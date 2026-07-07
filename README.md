# zenn_001

[Zenn](https://zenn.dev/) の記事・本を管理するリポジトリです。GitHub 連携でデプロイされます。

## 本

- `books/tts-from-text-to-audio/` — **TTS ― テキストが音になるまで**（全31章、無料）

## 記事

- `articles/first-openjtalk.md` — はじめてのOpenJTalk
- `articles/gpu-for-tts-training.md` — TTS学習向けのGPUの選び方
- `articles/tts-lineage-map-from-vits.md` — VITSから見るTTS 10系統マップ(2016–2026)
- `articles/japanese-tts-datasets.md` — 日本語TTSのためのデータセット選び

## ローカルプレビュー

```bash
npm install
npx zenn preview   # http://localhost:8000
```

## 新規記事・本

```bash
npx zenn new:article
npx zenn new:book
```

参考: [Zenn CLI で記事・本を管理する方法](https://zenn.dev/zenn/articles/zenn-cli-guide)
