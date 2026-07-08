---
title: "はじめに ― 全31章の読む順ガイド"
---

## この本について

本書は、TTS(音声合成)のしくみを、**1章1テーマ**で、図と最小限の数式だけで解きほぐす本です。各章は独立して読めますが、順番に読むと「テキストが音になるまで」の全体像と、その裏にある生成モデルの考え方、そして最前線までが一本につながるように作ってあります。

この章は**全31章の読む順ガイド**です。**3つの読むルート＋発展トピック**に分けて案内します。

:::message
各章の技術的な主張は、元論文を実際に読んで確認した内容に基づいています。図はすべて自作(matplotlib / mermaid)です。俯瞰には姉妹ページ「[VITSから見るTTS 10系統マップ(2016–2026)](https://zenn.dev/nnn112358/articles/tts-lineage-map-from-vits)」もどうぞ。
:::

## 全体マップ

```mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#475569','fontFamily':'Noto Sans CJK JP, sans-serif','fontSize':'15px'},'flowchart':{'padding':14,'nodeSpacing':42,'rankSpacing':48,'curve':'linear'}}}%%
flowchart LR
    A("ルートA<br/>パイプライン編<br/>テキスト→音の全工程"):::blue --> C("ルートC 総集編<br/>VITS / VITS2"):::amber
    B("ルートB<br/>生成モデル編<br/>VAE・Flow・GAN・MAS・SDP"):::purple --> C
    C --> D("最前線編<br/>StyleTTS 2 / LLM TTS /<br/>Qwen3-TTS / zero-shot"):::pink
    classDef blue fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827
    classDef purple fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#111827
    classDef amber fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#111827
    classDef pink fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#111827
```

## ルートA:パイプライン編(テキストが音になるまで)

TTSの処理の流れを、入口から出口まで順にたどるルートです。まずここから読むのがおすすめ。

| # | 章 | ひとこと |
|---|---|---|
| 1 | [G2P](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/g2p) | 文字を発音記号に変える、TTSの入口 |
| 2 | [音響モデル](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/acoustic-model) | 音素をメルに変える中核。アライメントが難所 |
| 3 | [メルスペクトログラム](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/mel-spectrogram) | 音の「設計図」となる中間表現 |
| 4 | [WaveNet](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/wavenet) | 波形を直接作る、ニューラルボコーダの元祖 |
| 5 | [HiFi-GAN](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/hifigan) | メルを音にするGANボコーダの決定版 |
| 6 | [iSTFTNet](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/istftnet) | 終盤をiSTFTに任せて軽く |
| 7 | [Vocos](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/vocos) | フーリエで一発、桁違いに速く |

## ルートB:生成モデル編(VITSの部品たち)

VITSを構成する生成モデルと部品を、1つずつ理解するルートです。

| # | 章 | ひとこと |
|---|---|---|
| 8 | [VAE](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/vae) | 分布に圧縮して生成する。VITSの骨格 |
| 9 | [Flow(正規化フロー)](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/flow) | 可逆変換でノイズを整形。VITSの"F" |
| 10 | [GAN](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/gan) | 偽造者vs鑑定士。VITSの"G" |
| 11 | [MAS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/mas) | 音素と音の対応を自力で探す |
| 12 | [SDP](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/sdp) | 毎回ちがう自然なリズムを作る |
| 13 | [Glow-TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/glow-tts) | FlowとMASが出会う、VITSの前身 |

## ルートC:総集編(合流点)

ルートA・Bの部品が、1つのモデルに合流します。

| # | 章 | ひとこと |
|---|---|---|
| 14 | [VITS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/vits) | VAE+Flow+GAN+MAS+SDPが1つになる場所 |
| 15 | [VITS2](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/vits2) | VITSの弱点を3つの改良でみがく |

## 最前線編(人間超えとLLMの時代)

| # | 章 | ひとこと |
|---|---|---|
| 16 | [StyleTTS 2](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/styletts2) | Style Diffusion + WavLMで人間超え |
| 17 | [BERT](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/bert) | 文脈を読んで抑揚を自然に |
| 18 | [LLM TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/llm-tts) | 音声を「言語モデル」で喋らせる路線 |
| 19 | [Qwen3-TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/qwen3-tts) | LLM×コーデックの超低遅延ストリーミング |
| 20 | [zero-shot TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/zero-shot) | 3秒の声で、学習していない人を喋らせる |

## 追補編(発展トピック)

パイプラインの理解が進んだら、より深い話題や歴史・最前線の要素技術へ。

| # | 章 | ひとこと |
|---|---|---|
| 21 | [Tacotron](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/tacotron) | 文字から音を直接作った、E2E TTSの原点 |
| 22 | [Tacotron 2](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/tacotron2) | Location-Sensitive Attention + WaveNetで人間レベルへ |
| 23 | [FastSpeech](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/fastspeech) | 非自己回帰で270倍高速、読み飛ばしゼロ |
| 24 | [VALL-E](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/valle) | 音声トークンの言語モデリングで zero-shot TTS を開拓 |
| 25 | [EnCodec](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/encodec) | 音声を離散トークンにするニューラルコーデック |
| 26 | [Flow Matching](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/flow-matching) | ノイズ→データへ「まっすぐ」進む生成 |
| 27 | [F5-TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/f5-tts) | 継続長予測もテキストエンコーダも無しで喋る |
| 28 | [Fish-Speech](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/fish-speech) | 72万時間×DualAR、RVQもG2Pも捨てたLLM TTS |
| 29 | [Grad-TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/grad-tts) | 拡散モデルで「ノイズからメルを彫り出す」TTS |
| 30 | [MaskGCT](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/maskgct) | マスク予測で完全非自己回帰の zero-shot TTS |

## こんな人はここから

- **とにかく全体像を知りたい** → ルートAを1→7へ。
- **VITSを理解したい** → ルートB(8→13)→ ルートC(14→15)。
- **最新のTTS事情を知りたい** → 16→20(前提が要るときだけ戻る)。
- **系譜・歴史が好き** → [VITSから見るTTS 10系統マップ](https://zenn.dev/nnn112358/articles/tts-lineage-map-from-vits)。

それでは、はじめましょう。🔊
