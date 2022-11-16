def lamkhongcong(inputSh=[1000,1000], outputSh=[32.7688852,100]):
    for i in outputSh:
        for y in inputSh:
            if i <=0 or y <=i or y <=0:
                print(f'inputShape must be > outputShape and value >= 1\ninputShape: {inputSh}  outputShape: {outputSh}')     
                exit()
    result = []
    a = inputSh[0]/outputSh[0]
    result.append(str(a).split('.')[0])
    a = inputSh[1]/outputSh[1]
    result.append(str(a).split('.')[0])
    value = int(result[0])*int(result[1])
    return value
arr = lamkhongcong([2440,1220])
print(f'so luong la: {arr}')
# ra 