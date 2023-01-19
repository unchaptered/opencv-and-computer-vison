[< Backwards](../../README.md)

# Deep Learning

- [x] Perceptron
    - [x] 1. Single Perceptron (AN : artificate neuron)
    - [x] 2. Multiple Perceptron (ANN : artificate neuron network)
    - [x] 3. Activate - Sigmoid Functions
    - [x] 4. Activate - Hyperbolic Tangent Functions
    - [x] 5. Activate - Rectified Linear Unit(ReLU)
    - [x] 6. What is Cost Function?
    - [x] 7. Cost - Quadratic Cost
    - [x] 8. Cost - Cross Entropy
    - [x] 9. Conclusion
- [x] Gradient Descent and BackPropagation

## Perceptron

대다수의 경우 Perceptron은 n개 이하의 요청에 1개의 결과를 반환합니다.

이러한 경우 Activate Functions의 알고리즘이 빛을 발하는데, 보편적으로 Hyperbolic Tangent Functions와 Rectified Linear Unit이 좋은 결과를 반환합니다.

1. [Perceptron, artificial neuron](./README.md#1-single-perceptron-an--artificate-neuron)
2. [Multiple Perceptron](./README.md#2-multiple-perceptron-ann--artificate-neuron-network)
3. [Activate - Sigmoid Functions](./README.md#3-activate---sigmoid-functions)
4. [Activate - Hyperbolic Tangent Functions](./README.md#4-activate---hyperbolic-tangent-functions)
5. [Activate - Rectified Linear Unit(ReLU)](./README.md#5-activate---rectified-linear-unitrelu)
6. [What is Cost Function?](./README.md#6-what-is-cost-function)
7. [Cost - Quadratic Cost](./README.md#7-cost---quadratic-cost)
8. [Cost - Cross Entropy](./README.md#8-cost---cross-entropy)
9. [Conclusion](./README.md#9-conclusion)

### 1. Single Perceptron (AN : artificate neuron)

[나무위키 / Perceptron](https://en.wikipedia.org/wiki/Artificial_neuron)은 n개 이하의 Inputs를 입력받아 Activate Functions를 실행시킵니다.

다음과 같은 Actviate Functions가 있다고 가정해보겠습니다.

AF는 모든 Inputs을 합산하여 결과값을 산출합니다.<br>
이 결과값이 양수라면 `1 혹은 Outputs 1`을 반환하고 음수라면 `0`을 리턴합니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Neuron3.svg/400px-Neuron3.svg.png)

간혹 결과값이 0이라면 Bias(편항값)을 결과값에 더해서 이를 해결합니다.

아래 시그마 연산식이 일반적인 Activate Functions이라고 했을 때, `+b`라는 편향값을 통해서 에러를 방지할 수 있습니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/8773ce7afe8e89c86732f0b59beb83f41f23c832)

### 2. Multiple Perceptron (ANN : artificate neuron network)

실제로는 Perceptron은 다수의 레이러 구조 하에서 작용합니다.

1. Input Layers
    - 실제 데이터의 값
    - 데이터를 입력값으로 받아들이는 레이어
2. Hidden Layers
    - Input Layers 와 Output Layers의 사이에 존재
    - 3개 이상의 Hidden Layers가 있으면 이를 Deep Network(심층 신경망)으로 불림
3. Output Layer
    - 출력값을 산출하는 레이어

![](https://static.javatpoint.com/tutorial/tensorflow/images/multi-layer-perceptron-in-tensorflow.png)

더 많은 레이어를 거칠수록 추상화 레벨이 올라가게 됩니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/be21980cc9e55ea0880327b9d4797f2a0da6d06e)

### 3. Activate - Sigmoid Functions

[#Perceptron (AN : artificate neuron)](./README.md#1-single-perceptron-an--artificate-neuron)에서 다룬 기본적인 Activate Functions의 고유한 한계는 Inputs의 총합이 -1 이거나 -1000이거나 `항상 0을 반환`하는 것에 있습니다.

따라서, 다음과 같은 [나무위키 / Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function)을 통해서 미세값이 반영된 Outputs을 얻을 수 있습니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/faaa0c014ae28ac67db5c49b3f3e8b08415a3f2b)

이를 그래프로 보면 다음과 같습니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Error_Function.svg/320px-Error_Function.svg.png)

### 4. Activate - Hyperbolic Tangent Functions

위에서 언급한 [#Sigmoid Functions](./README.md#3-activate---sigmoid-functions) 대신에 Hyperbolic Tangent Functions을 사용할 수 있습니다. 

이전까지는 1, 0을 반환하였지만 Sigmoid는 `1과 -1을 반환`합니다.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/c5339e8f573e8aa4082e7395089fb620a5ed3de1)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/cc9e4dd5d9c44875bd6dde6356b223e5cf18880c)

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/c6b439d4cd800e1c60c54e5d902c7fe5acb48302)

이를 그래프로 보면 다음과 같습니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Sinh_cosh_tanh.svg/220px-Sinh_cosh_tanh.svg.png)

### 5. Activate - Rectified Linear Unit(ReLU)

매우 일반적으로 쓰이는 사례이며, `max(O, z)` 을 통해서 연산처리 됩니다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/ReLU_and_GELU.svg/220px-ReLU_and_GELU.svg.png)

### 6. What is Cost Function?

Perceptron의 기능을 어떻게 평가하는지 알아보기 위해서, Cost Function을 사용해서 `예상값과 결과값이 얼마나 떨어져 있는지 확인`할 생각입니다.

사용되는 변수는 다음과 같습니다.

```bash
y = 실제값
a = Perceptron의 예상값
```

가중치 시간과 편향 관점에서...

```bash
w * x + b = z

# w * x = 입력값의 가중치 시간
# b = bias 편향값
```

위와 같이 연산처리된 z는 Activate Functions에 인자로 전달이 됩니다. 즉, Sigmoid Functions를 채택한 경우에는 시그마(z)는 a라는 결과값(에상값)이 될 것입니다.

```bash
σ(z) = a
# 시그마(z) = a
```

이렇게 계산된 Perceptron 예상값 a와 실제 결곽값 y를 비교하게 됩니다.

### 7. Cost - Quadratic Cost

Quadratic Cost Function은 2차 비용함수로 공식을 적용하는 과정에서 `성능 저하`가 약간 발생할 수 있습니다.

```bash
C = Σ (y - a)^2 / n

# y = 실제값
# a = 예상값
# n = 횟수
```

[PDF / "Proper Quadratic Cost Functions with an application to AT&T"](https://flora.insead.edu/fichiersti_wp/Inseadwp1988/88-22.pdf)

### 8. Cost - Cross Entropy

Cross Entropy Functions(교차 앤트로피 함수)는 더 빠른 학습이 가능하게 만들어줍니다.

y와 `a`의 가장 큰 차이점은 교차 엔트로피 비용 함수 덕분에 Perceptron의 학습속도가 매우 빠르다는 것입니다. 


```bash
C = (-1/n) * Σ(y*ln(a) + (1-y)*ln(1-a))
```

### 9. Conclusion

Multiple Percentron(ANN)이 효율적으로 정상 작동하기 위해서는 Activate Functions, Cost Functions을 잘 선택해야 합니다.

## Gradient Descent and Backpropagation

한글 제목 : 경사 하강법과 역전파

ANN을 이용하는 경우 다음과 같은 이슈를 맞이하게 됩니다.

1. 비용 최소화점 탐색
2. 오차 조절

이러한 경우에 각각 Gradient Descent와 Backpropagation을 사용할 수 있습니다.

1. [Gradient Descent](./README.md#1-gradient-descent경사-하강법)
2. [Backpropagation](./README.md#2-backpropagation역전파)

### 1. Gradient Descent(경사 하강법)

[Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)는 최솟값을 찾는데 최적 알고리즘입니다. 따라서, Cost Functions 최적화를 원하는 개발자들에게 큰 이점을 가지고 있습니다. 

```bash
# x 축 가중치(weight)
# y 축 비용(cost)
```

비용 최소화점을 찾기 위해서 가중치가 속한 부분을 `미분`하여 기울기를 연산합니다. 이 기울기가 `최솟값(0)`이 되는 방향까지 `가중치를 조절`하여 비용 최소화점을 찾을 수 있습니다.

![](https://static.javatpoint.com/tutorial/machine-learning/images/gradient-descent-in-machine-learning1.png)

최솟값 자체를 찾는 방식은 매우 간결하고 효율적이지만, 복잡한 그래프를 구성하고 있는 경우에는 다소 난해할 수 있습니다.

### 2. Backpropagation#역전파

Gradient Descent의 방법은 모든 가중치를 수정 및 조정 하는 경우에는 적합하지 않을 수 있습니다.

이러한 경우, Backpropagtion을 사용하여 연산이 종료된 outputs의 오차를 조절하게 됩니다.

단어 그대로, Percentron Network(ANN)을 거슬러 올라가면서 오류를 계산하는 방식입니다.
각 계층마다 Comparison(비교)를 하여 단계별로 조금씩 오류를 계산합니다.