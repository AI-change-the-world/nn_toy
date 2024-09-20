import 'layer.dart';
import 'loss/mse.dart';

class NeuralNetwork {
  List<Layer> layers;

  NeuralNetwork(this.layers);

  // 前向传播
  List<double> forward(List<double> inputs) {
    List<double> output = inputs;
    for (Layer layer in layers) {
      output = layer.forward(output);
    }
    return output;
  }

  // 反向传播
  void backward(
      List<double> predicted, List<double> actual, double learningRate) {
    // 计算输出层的误差：predicted - actual
    List<double> outputError =
        List.generate(predicted.length, (i) => predicted[i] - actual[i]);

    // 逐层进行反向传播
    for (int i = layers.length - 1; i >= 0; i--) {
      outputError = layers[i].backward(outputError, learningRate);
    }
  }

  // 训练：前向传播 + 反向传播
  void train(List<List<double>> inputs, List<List<double>> targets, int epochs,
      double learningRate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      double totalLoss = 0.0;

      for (int i = 0; i < inputs.length; i++) {
        // 前向传播
        List<double> predicted = forward(inputs[i]);

        // 计算损失
        totalLoss += mseLoss(predicted, targets[i]);

        // 反向传播
        backward(predicted, targets[i], learningRate);
      }

      print('Epoch $epoch, Loss: $totalLoss');
    }
  }
}
