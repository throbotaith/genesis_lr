# 🦿 Genesisでの脚式ロボティクス

[legged_gym](https://github.com/leggedrobotics/legged_gym) をベースとし、[Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main) 上で脚式ロボットを学習させるためのフレームワークです。

## 目次
- [🦿 Genesisでの脚式ロボティクス](#-genesisでの脚式ロボティクス)
  - [更新履歴](#更新履歴)
  - [特徴](#特徴)
  - [テスト結果](#テスト結果)
  - [インストール](#インストール)
  - [使い方](#使い方)
    - [クイックスタート](#クイックスタート)
    - [詳細な手順](#詳細な手順)
  - [Docker](#docker)
  - [ギャラリー](#ギャラリー)
  - [謝辞](#謝辞)
  - [TODO](#todo)

---
## 更新履歴

<details>
<summary>2025/03/22</summary>

- [legged_gym](https://github.com/lupinjia/legged_gym_ext) をベースにした新しいリポジトリを作成しました。

</details>

<details>
<summary>2025/02/10</summary>

- `measure_heights` をサポートし、外界情報を用いた歩行デモ ([go2_rough](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_rough)) を追加しました。

![](./test_resources//go2_rough_demo.gif)

</details>

<details>
<summary>2024/12/28</summary>

- 手順をまとめた [wiki](https://github.com/lupinjia/genesis_lr/wiki) を追加しました。

</details>

<details>
<summary>2024/12/26</summary>

- 地形機能を追加しました。選択できる地形タイプは `"plane"`, `"heightfield"` です。

  ![](./test_resources/terrain_demo.gif)

- テスト結果を [tests.md](./test_resources/tests.md) に移動しました。

</details>

<details>
<summary>2024/12/24</summary>

- 新しいデモ環境 `bipedal_walker` を追加しました。

</details>

<details>
<summary>2024/12/23</summary>

- `main` ブランチと `deploy` ブランチを分割しました。`deploy` ブランチはカスタム版 `rsl_rl` と一緒に使用してください（近日公開予定）。

</details>

---

## 特徴

- **[legged_gym](https://github.com/leggedrobotics/legged_gym) を完全に踏襲**

  `legged_gym` や `rsl_rl` に慣れている方であれば簡単に利用できます。

- **高速かつ省メモリ**

  4096 環境で平面歩行タスクを学習させた場合、Genesis 上での学習速度は [Isaac Gym](https://developer.nvidia.com/isaac-gym) と比べ約 **1.3 倍**、グラフィックメモリ使用量はおよそ **1/2** です。

  メモリ消費が少ないため、より多くの並列環境を動かすことができ、さらなる速度向上が期待できます。

## テスト結果

Genesis 上で行ったテスト結果は [tests.md](./test_resources/tests.md) を参照してください。

## インストール

1. Python>=3.10 の仮想環境を作成します。
2. [PyTorch](https://pytorch.org/) をインストールします。
3. [Genesis リポジトリ](https://github.com/Genesis-Embodied-AI/Genesis) の手順に従って Genesis をインストールします。
4. `rsl_rl` と `tensorboard` をインストールします。
   ```bash
   git clone git@github.com:leggedrobotics/rsl_rl.git
   cd rsl_rl && git checkout v1.0.2 && pip install -e . --use-pep517

   pip install tensorboard
   ```
5. `genesis_lr` をインストールします。
   ```bash
   git clone git@github.com:lupinjia/genesis_lr.git
   cd genesis_lr
   pip install -e .
   ```

## 使い方

### クイックスタート

デフォルトのタスクは `utils/helpers.py` 内で `go2` に設定されています。以下のコマンドで学習を開始できます。

```bash
cd legged_gym/scripts
python train.py --headless
```

学習後、`logs/go2` 以下の `run_name` を `go2_config.py` の `load_run` に貼り付けます。

![](./test_resources/paste_load_run.png)

その後 `play.py` を実行すると、学習済みモデルを可視化できます。

![](./test_resources/go2_flat_play.gif)

### Mini Pupper Maze 例

画像入力を用いた迷路ナビゲーションポリシーを学習するには次のようにします。

```bash
python train.py --env minipupper_maze_env --headless --timesteps 50000
```

Genesis ビューアを表示したい場合は `--headless` を外してください。

```bash
python train.py --env minipupper_maze_env --timesteps 50000
```

### 詳細な手順

詳細は [wiki ページ](https://github.com/lupinjia/genesis_lr/wiki) を参照してください。

## Docker

このリポジトリを `/home/teru/ws` にクローンした状態で、次のコマンドでイメージをビルドします。

```bash
docker build -t genesis-lr .
```

GPU を使用してコンテナを起動し、学習を開始する例です。ホストの `/home/teru/ws` をコンテナ内の同じ場所にマウントし、作業ディレクトリとして使用します。

```bash
docker run --gpus all -it \
  -v /home/teru/ws:/home/teru/ws \
  -w /home/teru/ws \
  genesis-lr bash
cd legged_gym/scripts
python train.py --headless
```

### Mini Pupper 2 RL を GUI 付きで実行

Genesis ビューアを表示する場合は X11 を許可し、`--headless` を付けずに実行します。

```bash
# ホスト側で
xhost +local:root
docker run --gpus all -it \
  --env DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/teru/ws:/home/teru/ws \
  -w /home/teru/ws \
  genesis-lr bash

# コンテナ内で
python train.py --env minipupper_maze_env --timesteps 50000
```

## ギャラリー

| Go2 | Bipedal Walker |
| --- | --- |
| ![](./test_resources/go2_flat_play.gif) | ![](./test_resources/bipedal_walker_flat.gif) |

## 謝辞

- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/main)
- [Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)

## TODO

- [x] ドメインランダム化の追加
- [x] 実機での検証
- [x] ハイトフィールド対応
- [x] `measure_heights` サポート
- [ ] go2 のデプロイデモと手順（通常版と外部推定器版）
- [ ] 教師あり・学習デモの追加
