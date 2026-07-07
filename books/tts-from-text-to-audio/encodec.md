---
title: "EnCodec ― 音声を「離散トークン」に変えるニューラルコーデック"
---

## この章について

[LLM TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/llm-tts)で、「音声を離散トークンにするコーデック」が主役級の部品として出てきました。この章はその代表格 **EnCodec**(2022, Meta)——**ニューラルコーデック**の話です。

EnCodec は、音声波形を **離散トークン(コード)** に圧縮し、そこから波形を復元するモデル。VALL-E をはじめ、LLM系TTSの「音声を言葉のように扱う」土台になっています。見ていきましょう。🗜️

:::message
EnCodec: Défossez et al., *"High Fidelity Neural Audio Compression"* (2022, [arXiv:2210.13438](https://arxiv.org/abs/2210.13438))。元をたどれば [VQ-VAE](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/vae) → SoundStream の系譜。本章の仕様は論文本文で確認しています。図は matplotlib と mermaid で作成しました。
:::

## 3行で言うと

- EnCodec = **音声波形を離散トークンに圧縮し、波形に戻す**ニューラルコーデック(エンコーダ+量子化+デコーダ)。
- キモは **RVQ(残差ベクトル量子化)**:粗い近似→残りを詰める…を繰り返し、少ないコードで高品質に。
- **ストリーミング・リアルタイム(CPU)** 動作。GAN由来の識別器で品質を上げ、TTS(VALL-E等)の"音声トークン"の定番になった。

## そもそもコーデックとは

コーデックは、音声や動画を**圧縮(encode)して、また元に戻す(decode)** 技術です。ネットで音楽や通話を流すとき、生の波形は重いので、**redundancy(冗長)を削って小さくする**。従来はMP3やOpusのような、人手で設計された信号処理でこれをやっていました。

EnCodec は、この圧縮・復元を **ニューラルネットに学習させた**もの。エンコーダ・量子化・デコーダを **端から端まで一括学習(end-to-end)** し、従来コーデックを上回る品質を、低ビットレートで達成しました。

## 3つの部品

```mermaid
%%{init: {'theme':'base','themeVariables':{'lineColor':'#475569','fontFamily':'Noto Sans CJK JP, sans-serif','fontSize':'15px'},'flowchart':{'padding':14,'nodeSpacing':46,'rankSpacing':50,'curve':'linear'}}}%%
flowchart LR
    WAV("音声波形"):::gray --> ENC("① エンコーダ<br/>(畳み込み + LSTM)<br/>波形→潜在ベクトル"):::blue
    ENC --> Q("② 量子化 RVQ<br/>潜在→離散トークン列"):::amber
    Q --> DEC("③ デコーダ<br/>トークン→波形"):::purple
    DEC --> OUT("復元した波形"):::green
    OUT -.->|"識別器で本物らしく<br/>(GAN由来)"| D("多解像度識別器"):::pink
    classDef blue fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827
    classDef amber fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#111827
    classDef purple fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#111827
    classDef green fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827
    classDef pink fill:#fce7f3,stroke:#db2777,stroke-width:2px,color:#111827
    classDef gray fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#111827
```

1. **エンコーダ**:波形を、畳み込み+LSTM で**低いフレームレートの潜在ベクトル**に圧縮(24kHz音声で毎秒75ステップ)。
2. **量子化(RVQ)**:潜在ベクトルを**離散トークン**に変換(次節)。ここが心臓部。
3. **デコーダ**:トークンから波形を復元(エンコーダの鏡像)。

学習は、波形の**再構成損失**(時間・周波数の両方)＋ [GAN](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/gan)由来の**多解像度識別器**による知覚的な損失で行います([→HiFi-GANのMSD/MRDと同じ発想](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/hifigan))。損失の重みを自動調整する **loss balancer** も工夫の一つ。

## 心臓部:RVQ(残差ベクトル量子化)

「連続な潜在ベクトルを、離散トークンにする」のが量子化です。素朴には、**コードブック(見本ベクトルの辞書)から一番近いものを選ぶ**([→VQ-VAE](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/vae))。でも1つの辞書で高品質を狙うと、辞書が巨大になりすぎます。

そこで **RVQ(Residual Vector Quantization / 残差ベクトル量子化)**。**まず粗く近似し、その"残差(ズレ)"を次のコードブックで詰め、さらにその残差を…** と多段で近づけます。

![EnCodecのRVQ:残差を多段で詰める](/images/encodec-rvq.png)
*目標ベクトル(★)に、1つ目のコードブックで大まかに近づき、残りのズレを2つ目・3つ目…で詰めていく。粗→細で、少数のコードブックでも高精度に量子化できる。段数(コードブック数)を変えるとビットレートを調整できる。*

RVQ の嬉しさは、**コードブックの段数でビットレートを可変**にできること。段を減らせば軽く(低品質)、増やせば高品質。EnCodec は 1.5〜24 kbps で state-of-the-art を達成しました。さらに小さな Transformer で**エントロピー符号化**すると、帯域を最大40%削減できます。

## なぜTTSで重要なのか

EnCodec が TTS で決定的に重要なのは、**音声を「離散トークンの列」に変える**からです。これで音声が、テキストの単語列と同じように扱えるようになり、**[LLMで音声を自己回帰生成する](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/llm-tts)** 道が開けました。

- **VALL-E**:EnCodec のトークンを [LLM](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/llm-tts) で生成 → 3秒 zero-shot クローン。
- **Bark / MusicGen / AudioGen** など、生成系オーディオの土台。

[系譜マップ](https://zenn.dev/nnn112358/articles/tts-lineage-map-from-vits)で言えば、**VQ-VAE → SoundStream → EnCodec → DAC** というコーデックの系譜。EnCodec の直接の親は **SoundStream**(RVQ+GANを確立)で、EnCodec はそれをより高品質・使いやすくしたものです。

## まとめ 🗜️

- EnCodec = **音声波形 ⇄ 離散トークン**のニューラルコーデック(エンコーダ + RVQ量子化 + デコーダ、end-to-end学習)。
- **RVQ(残差ベクトル量子化)**:粗い近似→残差を次々詰める。段数でビットレート可変、少コードで高品質。
- **ストリーミング・CPUリアルタイム**、GAN識別器で高音質。1.5〜24kbpsでSOTA。
- TTSでは**音声を"言葉のように"扱う鍵**。VALL-E など [LLM TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/llm-tts) の土台。VQ-VAE→SoundStream→EnCodec→DACの系譜。

「音を、単語のように離散化する」——この一手が、LLMで喋らせる時代の扉を開きました。

## 参考リンク

- [EnCodec (arXiv:2210.13438)](https://arxiv.org/abs/2210.13438) / [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- 関連する章: [LLM TTS](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/llm-tts) / [VAE(VQ-VAE)](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/vae) / [GAN](https://zenn.dev/nnn112358/books/tts-from-text-to-audio/viewer/gan) / [VITSから見るTTS 10系統マップ](https://zenn.dev/nnn112358/articles/tts-lineage-map-from-vits)
