import numpy 
import scipy.special # 시그모이드 함수 expit 사용을 위해 scipy.special 불러오기

################################

# 신경망 클래스의 정의 
class neuralNetwork:
  
  # 신경망 초기화하기
  def __init__(self,inputnodes,hiddennodes, outputnodes, learningrate):
    # 입력, 은닉, 출력 계층의 노드 갯수 설정
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes
    
    # 가중치 행렬 W_ih와 W_ho
  
    # 배열 내 가중치는 W_i_j로 표기. 노드 i에서 다음 계층의 노드 j로 연결됨을 의미
    # w11 w21
    # w12 w22 등    
    self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
    self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

    # 학습률
    self.lr = learningrate

    # 활성화 함수로는 시그모이드 함수를 이용
    self.activation_function = lambda x: scipy.special.expit(x)

    pass
  
  # 신경망 학습시키기
  def train(self,inputs_list,targets_list):

    # 입력 리스트를 2차원의 행렬로 변환
    inputs = numpy.array(inputs_list,ndmin=2).T
    targets = numpy.array(targets_list,ndmin=2).T

    # 은닉 계층으로 들어오는 신호를 계산
    hidden_inputs = numpy.dot(self.wih,inputs)
    # 은닉 계층에서 나가는 신호를 계산
    hidden_outputs = self.activation_function(hidden_inputs)


    # 최종 출력 계층으로 들어오는 신호를 계산
    final_inputs = numpy.dot(self.who, hidden_outputs)
    # 최종 출력 계층에서 나가는 신호를 계산
    final_outputs = self.activation_function(final_inputs)
    
    # 출력 계층의 오차는 (실제 값 - 계산 값)
    output_errors = targets - final_outputs
    # 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
    hidden_errors = numpy.dot(self.who.T,output_errors)

    # 은닉 계층과 출력 계층 간의 가중치 업데이트
    self.who += self.lr * numpy.dot((output_errors * final_outputs * 
                                 (1.0 - final_outputs)),numpy.transpose(hidden_outputs))

    # 입력 계층과 은닉 계층 간의 가중치 업데이트
    self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
                                 (1.0 -hidden_outputs)), numpy.transpose(inputs))
  
    pass
  
  # 신경망에 질의하기
  def query(self, inputs_list):
    # 입력리스트를 2차원 행렬로 변환
    inputs = numpy.array(inputs_list,ndmin = 2).T
    
    # 은닉 계층으로 들어오는 신호를 계산
    hidden_inputs = numpy.dot(self.wih,inputs)
    # 은닉 계층에서 나가는 신호를 계산
    hidden_outputs = self.activation_function(hidden_inputs)

    # 최종 출력 계층으로 들어오는 신호를 계산
    final_inputs = numpy.dot(self.who, hidden_outputs)
    # 최종 출력 계층에서 나가는 신호를 계산
    final_outputs = self.activation_function(final_inputs)
    
    return final_outputs
