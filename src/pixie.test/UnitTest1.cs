using System;
using System.Numerics;
using Xunit;

namespace pixie.test
{
    public class UnitTest1
    {
        [Fact]
        public void Test1()
        {
            var t = new[,]
            {
                { 2d, 3, 4 },
                { 4, 3 ,2 },
                { 5, 4 ,0 }
            }.ToTensor<double>();

            var activation = new Sigmoid();
            var x = activation.Compute(t);
            var d = activation.Derivative(t);
        }
    }
}
