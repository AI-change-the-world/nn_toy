import 'dart:math';

class BatchNorm {
  final int inputSize;
  double epsilon = 1e-5;
  late List<double> gamma;
  late List<double> beta;

  // 保存均值和方差
  late List<double> mean;
  late List<double> variance;

  BatchNorm(this.inputSize) {
    // 初始化 gamma 为 1, beta 为 0
    gamma = List.filled(inputSize, 1.0);
    beta = List.filled(inputSize, 0.0);
    mean = List.filled(inputSize, 0.0);
    variance = List.filled(inputSize, 0.0);
  }

  // 计算 batch 的均值
  List<double> calculateMean(List<List<double>> inputs) {
    List<double> mean = List.filled(inputSize, 0.0);
    int batchSize = inputs.length;

    for (var input in inputs) {
      for (int i = 0; i < inputSize; i++) {
        mean[i] += input[i];
      }
    }

    for (int i = 0; i < inputSize; i++) {
      mean[i] /= batchSize;
    }

    return mean;
  }

  // 计算 batch 的方差
  List<double> calculateVariance(List<List<double>> inputs, List<double> mean) {
    List<double> variance = List.filled(inputSize, 0.0);
    int batchSize = inputs.length;

    for (var input in inputs) {
      for (int i = 0; i < inputSize; i++) {
        variance[i] += pow(input[i] - mean[i], 2);
      }
    }

    for (int i = 0; i < inputSize; i++) {
      variance[i] /= batchSize;
    }

    return variance;
  }

  // 执行批量归一化的前向传播
  List<List<double>> forward(List<List<double>> inputs) {
    // 计算当前 batch 的均值和方差
    mean = calculateMean(inputs);
    variance = calculateVariance(inputs, mean);

    List<List<double>> normalized = [];
    for (var input in inputs) {
      List<double> normInput = List.filled(inputSize, 0.0);
      for (int i = 0; i < inputSize; i++) {
        // 归一化
        normInput[i] = (input[i] - mean[i]) / sqrt(variance[i] + epsilon);
        // 缩放和平移
        normInput[i] = gamma[i] * normInput[i] + beta[i];
      }
      normalized.add(normInput);
    }

    return normalized;
  }

  // 反向传播计算梯度
  List<List<double>> backward(List<List<double>> dOutputs,
      List<List<double>> inputs, double learningRate) {
    int batchSize = inputs.length;

    List<double> dGamma = List.filled(inputSize, 0.0);
    List<double> dBeta = List.filled(inputSize, 0.0);

    // 计算 dGamma 和 dBeta
    for (int i = 0; i < batchSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        dGamma[j] += dOutputs[i][j] *
            (inputs[i][j] - mean[j]) /
            sqrt(variance[j] + epsilon);
        dBeta[j] += dOutputs[i][j];
      }
    }

    // 更新 gamma 和 beta
    for (int i = 0; i < inputSize; i++) {
      gamma[i] -= learningRate * dGamma[i];
      beta[i] -= learningRate * dBeta[i];
    }

    // 计算输入的梯度
    List<List<double>> dInputs =
        List.generate(batchSize, (_) => List.filled(inputSize, 0.0));
    for (int i = 0; i < batchSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        dInputs[i][j] = gamma[j] * dOutputs[i][j] / sqrt(variance[j] + epsilon);
      }
    }

    return dInputs;
  }
}
