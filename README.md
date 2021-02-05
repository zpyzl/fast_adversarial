# fast_adversarial
fast adversarial on TextCNN

# epsilon的取值
由于词向量数据范围在-0.2~0.2左右，epsilon不能取相近的值，否则扰动让词向量变化太大，变成另外一个词了，达不到模型泛化的目的。所以epsilon取0.01，在词向量上很小的扰动，近似等于原来的词，但是泛化能力增强，不会因为一点误差就分类错误。

# PGD
参考https://zhuanlan.zhihu.com/p/91269728

正常训练：
反向传播，得到正常的grad

对抗训练：
在embedding上添加对抗扰动
反向传播，并在正常的grad基础上，累加对抗训练的梯度
恢复embedding参数
梯度下降，更新参数

# free
参考https://github.com/locuslab/fast_adversarial/tree/master/CIFAR10

    minibatch_replays = 8 # 每个epoch分为minibatch_replays次更新梯度
    epsilon = 0.01  
    lr_steps = config.num_epochs * len(train_iter) * minibatch_replays  # 因为每个epoch内更新minibatch_replays次梯度，所以总的梯度步数就是epoch*minibatch_replays
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    delta = torch.zeros(128, 1, 32, 300, dtype=torch.float) # delta以0为初始化，和输入数据的维度相同，以便在输入上相加
    delta.requires_grad = True # delta也需要梯度，因为是按梯度变化的
    for epoch in range(config.num_epochs):
        ...
        for i, (trains, labels) in enumerate(train_iter):
            for _ in range(minibatch_replays):
                outputs, delta = model(trains, delta)   # delta在下一个批量继续上一个批量的值
                loss = F.cross_entropy(outputs, labels)
                model.zero_grad()
                loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)  # 不让delta超出epsilon，否则扰动太大
                optimizer.step()
                delta.grad.zero_()
                scheduler.step()


# FGSM(fast adversarial方式)
主要区别是每次的训练，delta都是在-epsilon~epsilon之间初始化为随机值。
两个好处：
1.避免了每个epoch分为N步更新梯度的时间代价
2.保证了扰动微小

