List<double> leakyRelu(List<double> x, double alpha) {
  return x.map((xi) => xi > 0 ? xi : alpha * xi).toList();
}

List<double> leakyReluDerivative(List<double> x, double alpha) {
  return x.map((xi) => xi > 0 ? 1.0 : alpha).toList();
}
