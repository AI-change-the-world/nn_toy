import 'dart:math';

import 'activation_function/function.dart';

class Layer {
  late List<List<double>> weights; // 权重
  late List<double> biases; // 偏置
  late List<double> lastInput; // 前一层的输入
  late List<double> lastOutput; // 当前层的输出

  double alpha;

  int inputSize;
  int outputSize;

  Layer(this.inputSize, this.outputSize, {this.alpha = 0.1}) {
    Random random = Random();
    weights = List.generate(outputSize,
        (_) => List.generate(inputSize, (_) => random.nextDouble() - 1));
    biases = List.generate(outputSize, (_) => random.nextDouble() - 1);
  }

  // 前向传播
  List<double> forward(List<double> inputs) {
    lastInput = inputs;
    List<double> output = List.filled(outputSize, 0.0);

    for (int i = 0; i < outputSize; i++) {
      output[i] = biases[i];
      for (int j = 0; j < inputSize; j++) {
        output[i] += inputs[j] * weights[i][j];
      }
    }
    lastOutput = sigmoid(output);
    return lastOutput;
  }

  // 反向传播：计算并返回损失对输入的梯度
  List<double> backward(List<double> outputError, double learningRate) {
    List<double> inputError = List.filled(inputSize, 0.0);

    // 计算损失对每个权重的梯度
    for (int i = 0; i < outputSize; i++) {
      double dActivation = lastOutput[i] > 0 ? 1 : 0; // ReLU 导数
      double delta = outputError[i] * dActivation;

      // 更新偏置
      biases[i] -= learningRate * delta;

      // 计算输入误差并更新权重
      for (int j = 0; j < inputSize; j++) {
        inputError[j] += weights[i][j] * delta; // 输入误差传递到前一层
        weights[i][j] -= learningRate * delta * lastInput[j]; // 更新权重
      }
    }

    return inputError;
  }
}
