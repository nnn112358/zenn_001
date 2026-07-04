# zenn_001

[Zenn](https://zenn.dev/) の記事を管理するリポジトリです。GitHub 連携でデプロイされます。

## 記事

- `articles/first-openjtalk.md` — はじめてのOpenJTalk
- `articles/gpu-for-tts-training.md` — TTS学習向けのGPUの選び方
- `articles/tts-lineage-map-from-vits.md` — VITSから見るTTS 10系統マップ(2016–2026)
- `articles/what-is-mel-spectrogram.md` — メルスペクトログラムってなんだ？
- `articles/hifigan-for-cats.md` — 猫でもわかるHiFi-GAN
- `articles/g2p-for-cats.md` — 猫でもわかるG2P(下書き)

## ローカルプレビュー

```bash
npm install
npx zenn preview   # http://localhost:8000
```

## 新規記事

```bash
npx zenn new:article
```

参考: [Zenn CLI で記事・本を管理する方法](https://zenn.dev/zenn/articles/zenn-cli-guide)
