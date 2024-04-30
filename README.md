<div align="center">

 <div style="width: 80%; text-align: center; margin:auto;">
      <img style="width: 80%" src="logo.png">
</div> 
</div> 

<div>
  <h1>AI-Generated Image Quality Refiner</h1> 
</div>

<div>
  <a href="https://github.com/lcysyzxdxc" target="_blank">Chunyi Li</a><sup>1</sup>
  <a href="https://teowu.github.io" target="_blank">Haoning Wu</a><sup>2</sup>,
  <a href="https://hongkunhao.github.io/" target="_blank">Hongkun Hao</a><sup>1</sup>,
  <a href="https://github.com/zzc-1998" target="_blank">Zicheng Zhang</a><sup>1</sup>,
  <a href="https://github.com/QMME" target="_blank">Tengchuan Kou</a><sup>1</sup>,
  <a href="https://chaofengc.github.io" target="_blank">Chaofeng Chen</a><sup>2</sup>,
  <a href="https://jhc.sjtu.edu.cn/~xiaohongliu/" target="_blank">Xiaohong Liu</a><sup>1</sup>,
  <a href="http://leibai.site/" target="_blank">Lei Bai</a><sup>3</sup>,
  <a href="https://personal.ntu.edu.sg/wslin/Home.html" target="_blank">Weisi Lin</a><sup>2</sup>,
  <a href="https://ee.sjtu.edu.cn/en/FacultyDetail.aspx?id=24&infoid=153&flag=153" target="_blank">Guangtao Zhai</a><sup>1</sup>
</div>

<div>
  <sup>1</sup>Shanghai Jiao Tong University University, <sup>2</sup>Nanyang Technological University, <sup>3</sup>Shanghai AI Lab
</div>

_The official repo of AIGC image quality refiners:_

Q-Refine: Single optimization in perceptual quality.

G-Refine: General optimization in perceptual/alignment quality.

## 🔎Quality Map
### Perceptual Quality Map

```
python PQ-Map.py -p /orignal/image/path
```

Change `draw` to enable a quality map output. `multi` for using onr/multiple embedings. (Condisering overall quality, or quality in different aspect)

### Alignment Quality Map

## 🚀 Refining Code

### Q-Refine

### G-Refine

## 🌈Training

## 📌 TODO
- ✅ Release the PQ-Map code (Q-Refine and G-Refine)
- [ ] Release the AQ-Map code (G-Refine only)
- [ ] Release the Q-Refine code
- [ ] Release the G-Refine code
- [ ] Release the training script


## 📧 Contact
If you have any inquiries, please don't hesitate to reach out via email at `lcysyzxdxc@sjtu.edu.cn`

## 🎓Citations

If you find G-Refine is helpful, please cite:

```bibtex
@misc{G-Refine,
      title={G-Refine: A General Quality Refiner for Text-to-Image Generation}, 
      author={Chunyi Li and Haoning Wu and Hongkun Hao and Zicheng Zhang and Tengchaun Kou and Chaofeng Chen and Lei Bai and Xiaohong Liu and Weisi Lin and Guangtao Zhai},
      year={2024},
      eprint={2404.18343},
      archivePrefix={arXiv},
      primaryClass={cs.MM}
}
```

If you find Q-Refine is helpful, please cite:

```bibtex
@misc{Q-Refine,
      title={Q-Refine: A Perceptual Quality Refiner for AI-Generated Image}, 
      author={Chunyi Li and Haoning Wu and Zicheng Zhang and Hongkun Hao and Kaiwei Zhang and Lei Bai and Xiaohong Liu and Xiongkuo Min and Weisi Lin and Guangtao Zhai},
      year={2024},
      eprint={2401.01117},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
