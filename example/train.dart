import 'package:nn_toy/nn_toy.dart';

void main() {
  // 创建一个两层神经网络
  NeuralNetwork nn = NeuralNetwork([
    Layer(2, 3),
    Layer(3, 1),
  ]);

  // 示例训练数据：输入和目标输出
  List<List<double>> inputs = [
    [0.5, 0.8],
    [0.2, 0.3],
    [0.9, 0.7]
  ];

  List<List<double>> targets = [
    [0.6],
    [0.4],
    [0.9]
  ];

  // before train
  print(nn.forward([0.5, 0.8]));

  // 训练神经网络
  nn.train(inputs, targets, 1000, 0.01);

  // after train
  print(nn.forward([0.5, 0.8]));
}
