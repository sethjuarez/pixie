using System.Numerics;
using static System.Math;

namespace pixie
{
    public interface IActivation<T>
    {
        Tensor<T> Compute(Tensor<T> tensor);
        Tensor<T> Derivative(Tensor<T> tensor);
    }

    public class ReLU : IActivation<double>
    {
        public Tensor<double> Compute(Tensor<double> tensor)
        {
            var t = tensor.Clone();
            for(int i = 0; i < tensor.Length; i++)
                t.SetValue(i, tensor.GetValue(i) > 0 ? tensor.GetValue(i) : 0);

            return t;
        }

        public Tensor<double> Derivative(Tensor<double> tensor)
        {
            var t = tensor.Clone();
            for (int i = 0; i < tensor.Length; i++)
                t.SetValue(i, tensor.GetValue(i) > 0 ? 1 : 0);

            return t;
        }
    }

    public class Sigmoid : IActivation<double>
    {
        public Tensor<double> Compute(Tensor<double> tensor)
        {
            var t = tensor.Clone();
            for(int i = 0; i < tensor.Length; i++)
                t.SetValue(i, 1d / (1d + Exp(-tensor.GetValue(i))));

            return t;
        }

        public Tensor<double> Derivative(Tensor<double> tensor)
        {
            var t = tensor.Clone();
            for (int i = 0; i < tensor.Length; i++)
            {
                double e = 1d / (1d + Exp(-tensor.GetValue(i)));
                t.SetValue(i, e * (1 - e));
            }

            return t;
        }
    }
}