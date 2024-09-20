// ReLU 激活函数
import 'dart:math';

List<double> relu(List<double> inputs) {
  return inputs.map((x) => max(0, x).toDouble()).toList();
}
