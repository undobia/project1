# coding: utf-8
import sys, os

sys.path.append("D:\\CSC\\deep learning\\deep_learning_from_scratch")  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

#conv- ReLU - Pool - Affine - ReLU - Affine - Softmax
class SimpleConvNet:
    """단순한 합성곱 신경망

    conv - relu - pool - affine - relu - affine - softmax

    합성곱 층 (Conv1):
    입력에 대해 필터(커널)를 적용한 후, 편향을 더하여 특징 맵을 생성합니다.
    ReLU 활성화 (Relu1):
    합성곱 층의 출력에 ReLU 활성화 함수를 적용합니다.
    풀링 층 (Pool1):
    특징 맵의 크기를 줄여주는 풀링 연산을 적용합니다. (2x2 크기의 풀링)
    Affine(완전 연결) 층 (Affine1):
    이전 층의 출력과 모든 뉴런을 연결하는 완전 연결층입니다.
    ReLU 활성화 (Relu2):
    Affine 층의 출력에 ReLU 활성화 함수를 적용합니다.
    Affine 층 (Affine2):
    또 다른 완전 연결층입니다.
    Softmax 층 (SoftmaxWithLoss):
    출력 값을 확률로 변환하고, 손실 함수를 계산합니다. (크로스 엔트로피 손실)

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']   #필터 개수 : 30개
        filter_size = conv_param['filter_size'] # 3by3?, 5by5!
        filter_pad = conv_param['pad']  #padding을 0으로함 // 신경 안 쓰겠다는 뜻임
        filter_stride = conv_param['stride']    #stride : 1씩 이동
        input_size = input_dim[1]   #(1) : 28저장 28*28
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        #(28 - 5 + 2 * 0)/ (1 + 1) = 24, 합성곱후 출력크기는 24*24이다.
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        #30 * (24 / 2) * (24 / 2) = 30 * 12 * 12 = 4320
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        """
        filter_num: 첫 번째 합성곱 층의 필터 개수(예: 30개 필터).
        input_dim[0]: 입력 이미지의 채널 수(흑백 이미지인 경우 1, RGB 이미지인 경우 3).
        filter_size: 필터의 크기(예: 5x5).
        np.random.randn(...): 정규 분포를 따르는 난수를 생성합니다.
        여기서 filter_num, input_dim[0], filter_size, filter_size는 각 필터의 크기를 결정합니다.
        예를 들어, 필터가 30개이고, 각 필터의 크기가 5x5이며, 입력 이미지가 1채널(흑백)인 경우:
        W1.shape == (30, 1, 5, 5)
        즉, 첫 번째 합성곱 층의 가중치는 30 x 1 x 5 x 5 크기의 텐서입니다.
        """
        self.params['b1'] = np.zeros(filter_num)
        """
        첫 번째 합성곱 층의 편향을 0으로 초기화합니다.
        편향은 각 필터마다 하나씩 적용되므로, 편향의 크기는 filter_num에 해당하는 크기를 
        가집니다. 예를 들어, 필터가 30개라면, 편향도 30개의 값으로 구성됩니다.
        """
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        """
        pool_output_size: 풀링 후의 출력 크기입니다. 풀링 연산을 거쳐 이미지의 크기가 줄어든 후의 결과입니다. 계산결과 4320
        hidden_size: 은닉층의 뉴런 개수(예: 100개).
        이 가중치는 풀링 결과로 나온 크기와 은닉층의 뉴런 수를 연결하는 역할을 합니다.
        예를 들어, 풀링 후의 크기가 4320이라면, 이 층의 가중치는 4320 x 100 크기가 됩니다. 
        """
        self.params['b2'] = np.zeros(hidden_size)
        """
        첫 번째 완전 연결층(즉, Affine1)의 편향을 0으로 초기화합니다.
        은닉층에는 hidden_size 개수만큼의 뉴런이 있으므로, 편향도 hidden_size 크기를 가집니다.
        즉, b2는 100개의 값으로 구성됩니다.
        """
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        """
        hidden_size: 은닉층의 뉴런 수(예: 100개).
        output_size: 출력층의 뉴런 수(예: 10개 클래스).
        이 가중치는 은닉층의 출력과 출력층의 뉴런을 연결하는 역할을 합니다.
        예를 들어, 은닉층이 100개의 뉴런을 가지고 출력층이 10개의 뉴런이면,
         이 가중치는 100 x 10 크기를 가집니다.       
        """
        self.params['b3'] = np.zeros(output_size)
        """
        출력층의 편향을 0으로 초기화합니다.
        출력층의 크기는 output_size인 10이므로, 편향도 10개의 값으로 구성됩니다.
        """
        """
        이 코드는 각 층의 가중치(W1, W2, W3)와 편향(b1, b2, b3)을 초기화하는 작업을 수행하며, 이를 딕셔너리 self.params에 저장합니다. 가중치는 weight_init_std를 곱한 정규 분포 난수로 초기화하고, 편향은 모두 0으로 초기화합니다.

        W1: 첫 번째 합성곱 층의 가중치 (필터 개수 × 채널 수 × 필터 크기 × 필터 크기)
        b1: 첫 번째 합성곱 층의 편향 (필터 개수)
        W2: 첫 번째 완전 연결층(은닉층)의 가중치 (풀링 출력 크기 × 은닉층 뉴런 수)
        b2: 첫 번째 완전 연결층의 편향 (은닉층 뉴런 수)
        W3: 두 번째 완전 연결층(출력층)의 가중치 (은닉층 뉴런 수 × 출력 클래스 수)
        b3: 두 번째 완전 연결층의 편향 (출력 클래스 수)
        이렇게 초기화된 가중치와 편향은 네트워크의 학습 과정에서 업데이트되며, 점진적으로 더 나은 예측 성능을 얻기 위해 최적화됩니다.
        """


        # 계층 생성 conv - relu - pool - affine - relu - affine - softmax
        self.layers = OrderedDict()
        """
        OrderedDict(): Python의 기본 dict와 비슷하지만, 항목이 추가된 순서를 기억하는 딕셔너리입니다. 순서대로 연산이 이루어져야 하는 레이어를 설정할 때 유용합니다.
        이 구조를 사용하여 네트워크가 입력 데이터를 순차적으로 처리하도록 합니다.
        즉, Conv1 -> Relu1 -> Pool1 -> Affine1 -> Relu2 -> Affine2 -> SoftmaxWithLoss 순서로 연산이 진행됩니다.
        """
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        """
        Relu(): 합성곱 층의 출력에 ReLU 활성화 함수를 적용하여 음수를 모두 0으로 바꿉니다.
        ReLU 함수는 비선형성을 추가하여 네트워크가 복잡한 패턴을 학습할 수 있게 합니다.
        """
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        """
        Pooling: 풀링 층은 이미지 크기를 줄이면서 중요한 특징은 유지하는 역할을 합니다. 여기서는 2x2 크기의 풀링 윈도우를 사용하고, stride=2로 이동하여 출력 크기를 절반으로 줄입니다.
        풀링은 주로 **최대 풀링(Max Pooling)**을 사용하며, 이 경우 2x2 영역에서 최대값을 선택합니다.
        """
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        """
        Affine: 완전 연결층(fully connected layer)입니다. 이 층은 이전 층의 출력을 모두 연결하여 뉴런을 생성합니다. 여기서 **self.params['W2']**는 이 층의 가중치이며, **self.params['b2']**는 편향입니다.
        이 층은 뉴런 개수(hidden_size)가 설정된 은닉층으로, 특징을 종합하여 더 복잡한 패턴을 학습합니다.
        """
        self.layers['Relu2'] = Relu()
        """
        또 다시 ReLU 활성화 함수를 적용하여 은닉층에서 비선형성을 추가합니다. 이 과정은 네트워크가 복잡한 데이터 패턴을 학습하는 데 필요합니다.
        """
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        """
        Affine: 두 번째 완전 연결층으로, 출력층으로 이어지는 뉴런을 생성합니다. 이 층의 가중치는 **self.params['W3']**이고, 편향은 **self.params['b3']**입니다.
        이 층의 출력이 바로 네트워크의 최종 예측 값입니다. (여기서는 10개 클래스에 대한 확률을 출력)
        """
        self.last_layer = SoftmaxWithLoss()
        """
        SoftmaxWithLoss: 출력층으로, softmax 함수를 적용하여 각 클래스의 확률을 구하고, 손실을 계산하는 역할을 합니다. softmax 함수는 네트워크가 각 클래스에 대해 예측한 확률을 출력합니다.
        손실 함수: 출력된 확률과 실제 정답(label) 간의 차이를 계산하여 네트워크의 성능을 평가합니다.
        """
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        """
        predict 함수는 입력 데이터를 네트워크의 각 층을 통해 순차적으로 전달하여 최종 예측 값을 구하는 함수입니다.
        입력은 첫 번째 합성곱 층부터 마지막 출력층까지 순차적으로 처리되며, 각 층의 forward() 메서드가 호출됩니다.
        최종적으로 x가 출력으로 반환되며, 이는 네트워크의 예측 결과입니다.
        """
        return x



    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        """
        loss 함수는 네트워크의 예측 결과와 실제 정답 간의 차이를 계산하는 함수입니다.
        이 함수는 predict를 호출하여 예측 값을 구하고, 이를 마지막 층의 SoftmaxWithLoss를 통해 손실로 변환하여 반환합니다.
        손실 값은 네트워크가 얼마나 잘못 예측했는지, 즉 성능이 얼마나 좋은지를 평가하는 중요한 지표가 됩니다.
        """

        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        """
        t는 정답 레이블을 의미합니다. 만약 t가 원-핫 인코딩된 벡터로 되어 있다면, t.ndim은 2가 됩니다. (즉, 각 레이블이 0과 1로 된 벡터 형태로 저장됨)
        np.argmax(t, axis=1)는 원-핫 인코딩된 벡터를 정수 레이블로 변환하는 역할을 합니다. 예를 들어, [0, 0, 1, 0, 0]은 2로 변환됩니다.
        """
        acc = 0.0   #정확도를 저장할 변수

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
            """
            tx = x[i * batch_size:(i + 1) * batch_size]: 현재 배치의 입력 데이터를 슬라이싱합니다.
            tt = t[i * batch_size:(i + 1) * batch_size]: 현재 배치의 정답 레이블을 슬라이싱합니다.
            y = self.predict(tx): 현재 배치에 대해 예측을 수행합니다.
            y = np.argmax(y, axis=1): 예측된 결과 y는 각 클래스에 대한 확률로 되어 있는데, 이 중 가장 큰 값(가장 높은 확률을 가진 클래스)을 선택하여 정수 레이블로 변환합니다.
            acc += np.sum(y == tt): 예측 결과와 실제 정답 tt가 일치하는지 확인한 후, 일치하는 개수를 더합니다.
            """
        """
        accuracy 함수는 모델의 정확도를 계산하는 함수로, 배치 처리 방식을 사용하여 큰 데이터셋에 대해 효율적으로 정확도를 계산합니다.
        원-핫 인코딩된 레이블을 정수형 레이블로 변환하고, 배치 단위로 예측을 수행한 뒤, 정확한 예측 개수를 계산하여 전체 정확도를 반환합니다.
        정확도는 모델이 얼마나 잘 예측하는지의 비율을 나타내며, 모델의 성능을 평가하는 데 유용한 지표입니다.
        """
        """
        acc / x.shape[0]: 배치 단위로 계산한 정확도를 전체 데이터에 대해 계산합니다. 전체 정확도는 올바르게 예측한 샘플의 비율입니다.
        """
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    """
    params = {}
    먼저 빈 딕셔너리 params를 생성합니다. 이 딕셔너리에 신경망의 파라미터들을 저장합니다.
    
    for key, val in self.params.items():
    self.params는 신경망의 파라미터들을 담고 있는 딕셔너리입니다. 이 딕셔너리에는 가중치(W)와 편향(b)이 저장되어 있습니다. 예를 들어, W1, b1, W2, b2 등이 포함될 수 있습니다.
    items()는 딕셔너리에서 키와 값을 순차적으로 반환하므로, 이를 이용해 각 파라미터의 이름(key)과 값을(val) params 딕셔너리에 복사합니다.
    
    with open(file_name, 'wb') as f:
    file_name으로 지정된 파일을 바이너리 쓰기 모드('wb')로 엽니다. 만약 파일이 존재하지 않으면 새로 생성됩니다.
    file_name의 기본값은 "params.pkl"입니다. 즉, params.pkl이라는 파일로 저장됩니다.
    
    pickle.dump(params, f)
    pickle 모듈을 사용하여 params 딕셔너리를 바이너리 형식으로 파일에 저장합니다.
    pickle.dump() 함수는 파이썬 객체를 직렬화하여 파일에 저장하는 기능을 합니다. 여기서는 params 딕셔너리를 바이너리 형식으로 변환하여 파일 f에 저장합니다.
    """

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            """
            file_name으로 지정된 파일을 바이너리 읽기 모드('rb')로 엽니다. 기본 파일 이름은 "params.pkl"입니다.
            pickle.load(f)를 사용해 파일에서 저장된 파라미터를 읽어옵니다. 이때 params는 저장된 가중치(W)와 편향(b)를 포함한 딕셔너리입니다.
            """
        for key, val in params.items():
            self.params[key] = val
            """
            불러온 params 딕셔너리의 각 key와 val을 순차적으로 가져와서, 이를 모델의 파라미터(self.params)에 저장합니다.
            이렇게 하면 저장된 파라미터가 self.params에 업데이트되어, 네트워크의 가중치와 편향이 복원됩니다.
            """



        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]
    """
    이후, 각 레이어(Conv1, Affine1, Affine2)의 가중치(W)와 편향(b)를 업데이트합니다.
    예를 들어, self.layers['Conv1'].W는 self.params['W1']로 업데이트됩니다.
    i는 0부터 시작하는 인덱스를 이용해 'W1', 'W2', 'W3'와 같이 가중치 키를 생성하고 이를 self.layers[key].W에 할당합니다.
    """
    """
    load_params 함수는 저장된 파라미터 파일을 읽어와 신경망의 가중치와 편향을 복원하는 역할을 합니다.
    pickle 모듈을 사용해 파일에서 파라미터를 읽고, 이를 모델의 각 레이어에 적절히 할당하여 모델을 복원합니다.
    이를 통해, 학습한 모델을 저장하고 나중에 불러와서 예측이나 추가 학습을 계속할 수 있습니다.
    """