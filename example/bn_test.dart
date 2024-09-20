import 'package:nn_toy/src/nn/bn.dart';

void main() {
  BatchNorm batchNorm = BatchNorm(3);

  // 输入 batch (3 个特征，2 个样本)
  List<List<double>> inputs = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
  ];

  // 前向传播
  List<List<double>> normalized = batchNorm.forward(inputs);
  print("Normalized output: $normalized");

  // 假设这是来自损失函数的梯度
  List<List<double>> dOutputs = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
  ];

  // 反向传播
  List<List<double>> dInputs = batchNorm.backward(dOutputs, inputs, 0.01);
  print("Gradient with respect to inputs: $dInputs");
}
