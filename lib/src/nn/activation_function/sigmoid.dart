// Sigmoid 激活函数
import 'dart:math';

List<double> sigmoid(List<double> inputs) {
  return inputs.map((x) => 1 / (1 + exp(-x))).toList();
}
