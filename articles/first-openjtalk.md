---
title: "はじめてのOpenJTalk"
emoji: "🗣️"
type: "tech"
topics: ["openjtalk", "tts", "音声合成", "python"]
published: true
---

## OpenJTalk とは

**Open JTalk** は、名古屋工業大学を中心に開発されている**オープンソースの日本語テキスト音声合成(TTS)エンジン**です。テキストを入力すると日本語の音声(WAV)を合成できます。

技術的には次の2つを土台にしています。

- **HTS(HMM-based Speech Synthesis System)** … 隠れマルコフモデル(HMM)ベースの音声合成
- **MeCab** … 形態素解析(テキストを単語に分割し、読み・アクセントを推定)

ディープラーニング以前から使われてきた枯れた技術で、**軽量・完全オフライン・商用利用可(修正BSDライセンス)**という強みがあり、今でも「まず日本語をしゃべらせたい」という場面で重宝します。

:::message
最近の VITS などニューラルTTSと比べると声質は機械的ですが、**モデルが数MBと極小で、CPUだけで一瞬で動く**のが最大の利点です。
:::

## 仕組み(処理の流れ)

Open JTalk は内部で次のパイプラインを通ります。

```
テキスト
  │  ① テキスト解析(MeCab + 辞書)… 単語分割・読み・アクセント
  ▼
フルコンテキストラベル(音素+韻律の詳細情報)
  │  ② HTS … HMMから音響特徴量(メルケプストラム・F0)を生成
  ▼
音響特徴量
  │  ③ ボコーダ(MLSAフィルタ)… 波形を合成
  ▼
音声(WAV)
```

必要なものは大きく2つです。

- **辞書**(`open_jtalk_dic`)… テキスト解析用。NAIST日本語辞書ベース
- **音声モデル**(`.htsvoice`)… 声の正体。話者ごとに1ファイル

## インストール(Ubuntu / Debian)

`apt` で本体・辞書・標準音声をまとめて入れられます。

```bash
sudo apt update
sudo apt install -y \
  open-jtalk \
  open-jtalk-mecab-naist-jdic \
  hts-voice-nitech-jp-atr503-m001
```

インストールされる主なパスは次のとおりです。

| 種類 | パス |
|---|---|
| 辞書 | `/var/lib/mecab/dic/open-jtalk/naist-jdic` |
| 標準音声(男性) | `/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice` |

## コマンドラインでしゃべらせる

`echo` で渡したテキストを WAV に合成してみます。

```bash
echo "こんにちは。はじめてのオープンジェイトークです。" | open_jtalk \
  -x /var/lib/mecab/dic/open-jtalk/naist-jdic \
  -m /usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice \
  -ow output.wav
```

生成された `output.wav` を再生すれば、日本語が聞こえるはずです。

```bash
aplay output.wav      # ALSA の場合
# または
ffplay -nodisp -autoexit output.wav
```

### 主なオプション

| オプション | 意味 | 例 |
|---|---|---|
| `-x <dir>` | 辞書ディレクトリ | 必須 |
| `-m <file>` | 音声モデル(.htsvoice) | 必須 |
| `-ow <file>` | 出力WAVファイル | `-ow out.wav` |
| `-r <値>` | 話速(1.0が標準、大きいほど速い) | `-r 1.2` |
| `-fm <値>` | ピッチ(半音単位、+で高く) | `-fm 3` |
| `-a <値>` | 声質(オールパス係数、0〜1) | `-a 0.5` |
| `-g <値>` | 音量ゲイン(dB) | `-g 5` |

例えば「少し速く・高い声」にするなら:

```bash
echo "はやくちで しゃべります。" | open_jtalk \
  -x /var/lib/mecab/dic/open-jtalk/naist-jdic \
  -m /usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice \
  -r 1.3 -fm 4 -ow fast.wav
```

:::message
女性の声を使いたい場合は、MMDAgent 配布の **「メイ(mei)」音声**(`mei_normal.htsvoice` など)が有名です。`.htsvoice` を入手して `-m` に指定するだけで差し替えられます。
:::

## Python から使う(pyopenjtalk)

コマンドラインより実用的なのが、Pythonラッパーの **`pyopenjtalk`** です。辞書や音声モデルを内蔵しているので、パス指定なしで即使えます。

```bash
pip install pyopenjtalk
```

### 音声合成

```python
import numpy as np
import pyopenjtalk
from scipy.io import wavfile

x, sr = pyopenjtalk.tts("こんにちは、世界。")
wavfile.write("out.wav", sr, x.astype(np.int16))
```

### 読み(音素列)を取り出す:G2P

```python
import pyopenjtalk

print(pyopenjtalk.g2p("今日はいい天気ですね"))
# k y o o w a i i t e N k i d e s U n e

# カタカナで欲しいとき
print(pyopenjtalk.g2p("今日はいい天気ですね", kana=True))
# キョーワイイテンキデスネ
```

### フルコンテキストラベル

音素だけでなく、アクセントや位置情報を含む**フルコンテキストラベル**も取れます。

```python
import pyopenjtalk

labels = pyopenjtalk.extract_fullcontext("こんにちは")
print(labels[0])
```

## ニューラルTTSとの関係

「HMMベースなのに今さら?」と思うかもしれませんが、**Open JTalk(pyopenjtalk)は現代のニューラル日本語TTSの前段としても現役**です。

VITS などを日本語で学習するとき、**テキスト→音素・アクセント**の変換部分に pyopenjtalk がよく使われます。ESPnet など主要なTTSツールキットも、日本語の g2p に pyopenjtalk を採用しています。

```
テキスト ──[pyopenjtalk g2p]──▶ 音素+アクセント ──[VITS等]──▶ 高品質音声
          (Open JTalkの解析部)                (ニューラルボコーダ)
```

つまり Open JTalk は、**単体の軽量TTS**としても、**最新TTSの日本語フロントエンド**としても使える、日本語音声合成の基礎技術です。

## まとめ

- Open JTalk は **HMMベースの軽量・オフライン日本語TTS**
- Ubuntu なら `apt` で本体・辞書・音声を入れ、`open_jtalk` コマンドで即合成できる
- Python なら **`pyopenjtalk`** が手軽で、音声合成・G2P・フルコンテキストラベルまで一通りできる
- 現代のニューラル日本語TTSの**フロントエンド**としても広く使われている

まずは `apt install` して1文しゃべらせるところから始めてみてください。
