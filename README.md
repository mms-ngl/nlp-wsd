## Word Sense Disambiguation

Word Sense Disambiguation (WSD) project for "Natural Language Processing" course.

### 📝 Project documentation

[**REPORT**](https://github.com/mms-ngl/nlp-wsd/blob/main/report.pdf)

### Course Info: http://naviglinlp.blogspot.com/

### 🚀 Project setup

#### Project directory
[[Downloads]](https://drive.google.com/drive/folders/1SEdDLFvQoapy0_jED9_OpE3JJzXPYKB8?usp=sharing) Label vocabulary and trained model: label2id.pth, model_weights.pth.
```
root
- data
- logs
- model
 - label2id.pth
 - model_weights.pth
 - .placeholder
- wsd
- Dockerfile
- README
- report
- requirements
- test
```

#### Requirements

* Ubuntu distribution
  * Either 20.04 or the current LTS (22.04).
* Conda 

#### Setup Environment

To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

#### Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

#### Setup Client

```bash
conda create -n nlp-wsd python=3.9
conda activate nlp-wsd
pip install -r requirements.txt
```

#### Run

*test.sh* is a simple bash script. To run it:

```bash
conda activate nlp-wsd
bash test.sh data/coarse-grained/test_coarse_grained.json
```
