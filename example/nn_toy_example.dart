import 'package:nn_toy/nn_toy.dart';

void main() {
  // 创建一个两层神经网络，第一层有 2 个输入和 3 个输出，第二层有 3 个输入和 1 个输出
  NeuralNetwork nn = NeuralNetwork([
    Layer(2, 10),
    Layer(10, 1),
  ]);

  // 示例输入
  List<double> inputs = [0.5, 0.8];

  // 前向传播输出
  List<double> output = nn.forward(inputs);

  print('网络输出: $output');
}
