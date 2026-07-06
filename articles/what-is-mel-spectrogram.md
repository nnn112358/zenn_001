---
title: "猫でもわかるメルスペクトログラム ― TTSの中間表現を図で理解する"
emoji: "📊"
type: "tech"
topics: ["音声合成", "tts", "音声処理", "python", "機械学習"]
published: false
---

## この記事について

音声合成(TTS)や音声認識の論文・実装を読むと、ほぼ必ず **「メルスペクトログラム(mel spectrogram)」** が出てきます。TacotronもFastSpeechも、出力するのはメルスペクトログラム。HiFi-GANのようなボコーダが受け取るのも、メルスペクトログラムです。

でも「メルってなに？」「スペクトログラムと何が違うの？」は意外と説明されないまま話が進みがちです。この記事では、**何を・なぜ・どうやって作るのか**を、実際に生成した図とコードで最短ルートで理解します。

:::message
この記事の図は、外部の音声ファイルを使わず、**合成した「音声っぽい信号」から numpy + matplotlib だけで生成**しています。ライブラリのブラックボックスではなく、中で何が起きているかが分かるように作りました。
:::

## 3行で言うと

- **スペクトログラム** = 音を「時間 × 周波数 × 強さ」の画像にしたもの。
- **メル** = その周波数軸を、**人間の耳の感度に合わせて歪めた(低音を細かく・高音を粗く)** もの。
- TTSでは、波形そのままだと重すぎ・扱いにくいので、この**コンパクトな中間表現**を音響モデルとボコーダの「受け渡し規格」として使う。

## まず「スペクトログラム」から

音の波形(waveform)は、1秒あたり22,050個といった大量の数字の列です。人間にとっても機械にとっても、これを直接眺めても「どんな音か」は分かりません。

そこで、音を短い時間ごとに区切って **「その瞬間にどの周波数成分がどれくらい含まれるか」** を計算し、画像として並べたものが **スペクトログラム** です。横軸が時間、縦軸が周波数、色の明るさがその強さ(エネルギー)を表します。

下図の (1)→(2) を見てください。同じ2.6秒の音を、波形とスペクトログラムで表したものです。

![波形・線形スペクトログラム・メルスペクトログラムの比較](/images/mel-pipeline.png)
*同じ音声クリップの3つの表現。(1)波形 → (2)線形スペクトログラム → (3)メルスペクトログラム。*

波形(1)ではただの「塊」にしか見えなかったものが、スペクトログラム(2)では構造が見えます。

- **横方向の縞** = 倍音(harmonics)。声の基本周波数(ピッチ)の整数倍にエネルギーが並ぶ、母音のような有声音の特徴。
- **右肩上がり・下がりの傾き** = ピッチが時間とともに動いている(抑揚)。
- **縦方向のモヤっとした帯**(0.8秒, 1.6秒あたり) = 広い周波数に散らばるノイズ状の成分。無声子音(「s」「f」のようなかすれ音)に相当。

## なぜ波形そのままではダメなのか

「じゃあ波形をそのまま学習すればいいのでは？」という疑問には、実用上の理由があります。

- **系列が長すぎる**: 1秒 = 22,050サンプル。数秒の発話で数万〜数十万点。時系列モデルには重い。
- **情報が密すぎ・冗長**: 波形は位相まで含む生データ。音色を捉えるには「どの周波数が強いか」で十分なことが多い。
- **予測が難しい**: 波形を直接予測させると学習が不安定。スペクトログラムのような**滑らかで低次元な表現**の方が、音響モデルにとって予測しやすい。

スペクトログラムは、時間方向を数百フレーム、周波数方向を数百〜数十次元に圧縮した、**扱いやすい「音の画像」** なのです。

## 「メル」= 人間の耳に合わせた周波数の物差し

線形スペクトログラム(2)の縦軸は、0Hzから11kHzまで**等間隔**です。しかし人間の聴覚は、そんなふうには周波数を感じていません。

**低い音の違いには敏感で、高い音の違いには鈍感** です。100Hzと200Hzは「1オクターブ違う別の音」に聞こえますが、8000Hzと8100Hzはほとんど区別できません。

この「人間の音高知覚」を近似したのが **メル尺度(mel scale)** です。変換式は次の通り(HTK式):

$$
m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)
$$

$f$ が物理周波数[Hz]、$m$ がメル値。対数なので、**低周波は引き伸ばされ(細かく)、高周波は圧縮されます(粗く)**。

![メルフィルタバンクとメル尺度](/images/mel-filterbank.png)
*左: メルフィルタバンク(80本中16本を表示)。低域では三角フィルタが狭く密、高域では広く疎になる。右: メル尺度の変換曲線。メル軸で等間隔に取ると、Hz軸では低域が密・高域が疎になる。*

スペクトログラムを「メル化」するとは、線形周波数の各成分を、この**メル尺度上で等間隔に並んだ三角形のフィルタ群(メルフィルタバンク)** でまとめ直すこと。上図左の三角フィルタひとつひとつが、メルスペクトログラムの1チャンネル(1行)になります。

結果、周波数軸が513次元(線形)から **80次元(メル)** といった具合に圧縮され、しかも人間の知覚に近い並びになる。これが図(2)→(3)の変化です。倍音の縞が下(低メルチャンネル)に密集し、全体がコンパクトになっているのが分かります。

## 作り方(計算パイプライン)

メルスペクトログラムは、次の手順で計算されます。

```mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#475569','fontFamily':'Noto Sans CJK JP, sans-serif','fontSize':'15px'},'flowchart':{'padding':14,'nodeSpacing':50,'rankSpacing':60,'curve':'linear'}}}%%
flowchart LR
    A("波形<br/>(1次元)"):::gray
    B("フレーム分割<br/>+ 窓関数"):::blue
    C("FFT<br/>(STFT)"):::blue
    D("振幅→パワー<br/>∣X∣²"):::blue
    E("メルフィルタバンク<br/>を掛ける"):::blue
    F("log を取る<br/>(log-mel)"):::blue
    G("メルスペクトログラム<br/>(n_mels × フレーム数)"):::green
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    classDef blue fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827
    classDef amber fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#111827
    classDef purple fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#111827
    classDef pink fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#111827
    classDef green fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827
    classDef gray fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827
    classDef red fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#111827
```

1. **フレーム分割 + 窓関数**: 波形を `win_length` サンプルずつの短い窓に切り、`hop_length` ずつずらしていく。窓の境界の不連続を抑えるためハン窓などを掛ける。
2. **STFT(短時間フーリエ変換)**: 各フレームをFFTして周波数成分に分解。`n_fft/2 + 1` 本の周波数ビンが得られる。
3. **パワー化**: 複素スペクトルの振幅の2乗を取る(**位相はここで捨てられる** ← 後述の重要点)。
4. **メルフィルタバンク適用**: `n_mels` 本の三角フィルタを掛けて周波数軸をメルに圧縮。
5. **log**: 人間の音量知覚(対数的)に合わせ、対数を取る。これで **log-mel spectrogram** の完成。

## 主要パラメータ

実装で必ず出てくるパラメータたちです。値は22.05kHz音声でのTTSの定番例。

| パラメータ | 意味 | 定番値 | 効果 |
|---|---|---|---|
| `sample_rate` | サンプリング周波数 | 22050 / 24000 | 扱える最高周波数 = この半分 |
| `n_fft` | FFT窓のサイズ | 1024 | 大きいほど周波数分解能↑・時間分解能↓ |
| `win_length` | 窓関数の長さ | = n_fft | 通常 n_fft と同じ |
| `hop_length` | フレームのずらし幅 | 256 | 小さいほど時間分解能↑・データ量↑ |
| `n_mels` | メルチャンネル数 | 80 | 出力の縦の次元。TTSは80が定番 |
| `fmin` / `fmax` | メルの下限/上限周波数 | 0 / sr/2 | 声の帯域に絞ることも(例: 0〜8000) |

`hop_length=256`, `sr=22050` なら、1秒あたり `22050 / 256 ≈ 86` フレーム。図(3)の横方向の細かさはこれで決まります。

## コードで作る

### librosa

いちばん手軽なのは [librosa](https://librosa.org/) です。

```python
import librosa
import numpy as np

y, sr = librosa.load("speech.wav", sr=22050)      # 波形を読み込み
mel = librosa.feature.melspectrogram(
    y=y, sr=sr,
    n_fft=1024, hop_length=256,
    n_mels=80, fmin=0, fmax=sr // 2,
)                                                  # (80, フレーム数) のパワーメル
mel_db = librosa.power_to_db(mel, ref=np.max)      # dB スケールに変換して表示用
print(mel.shape)  # 例: (80, 224)
```

### torchaudio

学習パイプラインに組み込むなら [torchaudio](https://pytorch.org/audio/) の変換が便利です(GPU・バッチ対応)。

```python
import torchaudio
import torchaudio.transforms as T

wav, sr = torchaudio.load("speech.wav")            # (channel, samples)
mel = T.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024, hop_length=256,
    n_mels=80, f_min=0, f_max=sr // 2,
)(wav)                                             # (channel, 80, フレーム数)
mel_db = T.AmplitudeToDB(stype="power")(mel)
```

### numpy だけで自作(中身を理解する)

ライブラリの中身は、実はこれだけです。メルフィルタバンクの構築が核心。

```python
import numpy as np

def hz_to_mel(f):  return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_bins)
    # メル軸で等間隔に n_mels+2 点 → Hz に戻す
    m_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, ce, hi = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        left  = (fft_freqs - lo) / (ce - lo)     # 立ち上がり
        right = (hi - fft_freqs) / (hi - ce)     # 立ち下がり
        fb[i] = np.clip(np.minimum(left, right), 0, None)  # 三角形
    return fb

# power: (n_bins, フレーム数) の STFT パワースペクトル
# mel = fb @ power  →  (n_mels, フレーム数)
```

この記事の図も、まさにこのフィルタバンク関数で生成しています。

## メルスペクトログラムの読み方

もう一度、図(3)を「読める」ようになっておきましょう。

- **横軸** = 時間、**縦軸** = メルチャンネル(下=低音、上=高音)、**色** = そのマスのエネルギー(明るいほど強い)。
- **下の方に並ぶ横縞** = 倍音構造。有声音(母音)のサイン。
- **特定の高さの明るい帯** = フォルマント(声道の共鳴。母音の種類を決める)。図では中央付近(mel 40〜45)に強い帯が見えます。
- **縦に伸びるモヤ** = 無声子音。倍音構造を持たず、広い帯域に散る。
- **縞の上下動** = ピッチ(抑揚)の変化。

TTSの音響モデルは、テキストからまさにこの「模様」を予測しているわけです。

## TTSにおける位置づけ ― 2段構成の「境界」

TTSの多くは **2段構成** で、メルスペクトログラムはその**受け渡し規格**です。

```mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#475569','fontFamily':'Noto Sans CJK JP, sans-serif','fontSize':'15px'},'flowchart':{'padding':14,'nodeSpacing':50,'rankSpacing':60,'curve':'linear'}}}%%
flowchart LR
    T("テキスト"):::gray
    AM("音響モデル<br/>(Tacotron2 / FastSpeech2)"):::blue
    VO("ボコーダ<br/>(HiFi-GAN など)"):::blue
    W("波形"):::green
    T --> AM
    AM -->|"メルスペクトログラム"| VO
    VO --> W
    classDef blue fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827
    classDef amber fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#111827
    classDef purple fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#111827
    classDef pink fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#111827
    classDef green fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827
    classDef gray fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827
    classDef red fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#111827
```

- **音響モデル**(Tacotron 2, FastSpeech 2 など)= テキスト → メルスペクトログラム
- **ボコーダ**(HiFi-GAN, BigVGAN, Vocos など)= メルスペクトログラム → 波形

この「境界」があるおかげで、音響モデルとボコーダを別々に開発・差し替えできます。

:::message
一方で **VITS** のような単段(End-to-End)モデルは、メルスペクトログラムを外に出さず、VAEの潜在変数として内部に閉じ込めてしまいます。「メルという中間表現をあえて捨てた」のがVITS系の特徴とも言えます。TTSの系譜については別記事「[VITSから見るTTS 10系統マップ](https://zenn.dev/nnn112358/articles/tts-lineage-map-from-vits)」で整理しています。
:::

## ハマりどころ・よくある疑問

- **位相はどこへ？**: パワー化(手順3)の時点で位相情報は捨てられます。だから**メルスペクトログラムから波形へは一意に戻せない**。位相を復元する必要があり、それが Griffin-Lim(反復推定)やニューラルボコーダ(HiFi-GAN等)の役割です。「なぜボコーダが要るのか」の答えがこれ。
- **`log` の中身**: `log(0)` を避けるため、実装では `log(mel + eps)` や `power_to_db` を使います。学習時は平均・分散で正規化するのも定番。
- **`n_mels` はなぜ80？**: 経験的にTTS品質と計算量のバランスが良い値。論文により 80 / 100 / 128 など。ボコーダと音響モデルで**必ず同じ設定に揃える**必要があります(ここがズレると全く鳴らない、が初学者の頻出ミス)。
- **メル vs MFCC**: MFCC(メル周波数ケプストラム係数)は、log-melにさらにDCTを掛けて次元圧縮したもの。音声認識の古典的特徴量ですが、現代のTTS/ニューラル系は**log-melをそのまま**使うことが多いです。

## 猫のまとめ 📊

- メルスペクトログラム = **「時間 × メル周波数 × 対数エネルギー」の音の画像**。
- **メル尺度**で周波数軸を人間の聴覚に合わせて歪め、`n_mels`(例:80)次元にコンパクト化している。
- 作り方は `波形 → 窓かけ → STFT → パワー → メルフィルタバンク → log` の一本道。
- TTSでは**音響モデルとボコーダをつなぐ中間表現**。位相を捨てているため、波形に戻すにはボコーダが要る。

「メルスペクトログラム」と聞いて身構えていた人も、これで論文や実装のコードが読めるはずです。

## 参考リンク

- [librosa `melspectrogram` ドキュメント](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html)
- [torchaudio `MelSpectrogram` ドキュメント](https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html)
- 関連記事: [VITSから見るTTS 10系統マップ(2016–2026)](https://zenn.dev/nnn112358/articles/tts-lineage-map-from-vits)

:::message
🐾 **猫でもわかるTTSシリーズ**(全32本) ― [目次](https://zenn.dev/nnn112358/articles/tts-for-cats-index) ／ 前: [音響モデル](https://zenn.dev/nnn112358/articles/acoustic-model-for-cats) ／ 次: [WaveNet](https://zenn.dev/nnn112358/articles/wavenet-for-cats)
:::
