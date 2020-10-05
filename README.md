# detectmasks
カレントディレクトリに"weights"というディレクトリを作成し、以下のURLに載っているweightファイルをダウンロードしてください。  
https://drive.google.com/drive/folders/1syqTIXZdG3MhS99Xv7zL-f-Gf82q0FmZ?usp=sharing  
ダウンロード後に、以下のコマンドで検知プログラムを実行できるようになります。  
```python
python plot_mask_DSFD-ResNet_withSoftmax.py pictures_directory sample.json False
```

本研究で用いられているCNNのうち、1段階目の顔検出で用いている学習済みのDSFDは以下のgitから引用しました。   
https://github.com/cs-giung/face-detection-pytorch
