// 均方误差（MSE）损失函数
import 'dart:math';

double mseLoss(List<double> predictions, List<double> targets) {
  double sum = 0.0;
  for (int i = 0; i < predictions.length; i++) {
    sum += pow(predictions[i] - targets[i], 2);
  }
  return sum / predictions.length;
}
