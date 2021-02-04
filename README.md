# fast_adversarial
fast adversarial on TextCNN

# FGSM(fast adversarial方式)
一开始用的epsilon是0.25，结果precision和recall都是1。这是因为预训练的词向量数据大小在0.2左右，加上-0.25~0.25之间的扰动，变化太大，变成另外一个词了。

所以我把epsilon改成0.025：

Iter:   8500,  Train Loss:  0.96,  Train Acc: 47.27%,  Val Loss:  0.92,  Val Acc: 49.65%,  Time: 0:07:16
Iter:   8600,  Train Loss:  0.86,  Train Acc: 48.44%,  Val Loss:  0.92,  Val Acc: 49.73%,  Time: 0:07:21
Iter:   8700,  Train Loss:  0.83,  Train Acc: 49.61%,  Val Loss:  0.92,  Val Acc: 49.65%,  Time: 0:07:26
Iter:   8800,  Train Loss:   0.9,  Train Acc: 50.78%,  Val Loss:  0.92,  Val Acc: 49.72%,  Time: 0:07:31
Iter:   8900,  Train Loss:  0.87,  Train Acc: 50.78%,  Val Loss:  0.92,  Val Acc: 49.80%,  Time: 0:07:37
Iter:   9000,  Train Loss:   0.9,  Train Acc: 47.66%,  Val Loss:  0.92,  Val Acc: 49.61%,  Time: 0:07:42
Iter:   9100,  Train Loss:  0.91,  Train Acc: 52.34%,  Val Loss:  0.92,  Val Acc: 49.76%,  Time: 0:07:47
No optimization for a long time, auto-stopping...
Test Loss:  0.91,  Test Acc: 49.79%
Precision, Recall and F1-Score...
               precision    recall  f1-score   support

      finance     0.4966    0.3660    0.4214      1000
       realty     0.4922    0.3800    0.4289      1000
       stocks     0.4975    0.2980    0.3727      1000
    education     0.4974    0.4870    0.4922      1000
      science     0.4845    0.3750    0.4228      1000
      society     0.4994    0.3950    0.4411      1000
     politics     0.4872    0.3620    0.4154      1000
       sports     0.4960    0.4340    0.4629      1000
         game     0.4942    0.3410    0.4036      1000
entertainment     0.4994    0.4240    0.4586      1000
    perturbed     0.5000    0.6095    0.5493     10000

     accuracy                         0.4979     20000
    macro avg     0.4949    0.4065    0.4426     20000
 weighted avg     0.4972    0.4979    0.4906     20000

Confusion Matrix...
[[ 366    2    1    0    0    0    0    0    0    0  631]
 [   1  380    0    0    0    0    0    0    0    0  619]
 [   2    1  298    0    4    0    3    0    0    0  692]
 [   0    0    0  487    0    0    1    0    0    0  512]
 [   0    0    0    0  375    1    1    0    5    0  618]
 [   0    3    0    2    1  395    1    0    0    0  598]
 [   0    0    3    1    1    0  362    0    0    0  633]
 [   0    0    0    0    0    0    0  434    0    1  565]
 [   0    0    0    0    6    0    0    0  341    0  653]
 [   0    0    0    0    0    0    0    1    0  424  575]
 [ 368  386  297  489  387  395  375  440  344  424 6095]]
 
 precision不到0.5，说明产生了很大的干扰，模型无法识别对抗攻击。
