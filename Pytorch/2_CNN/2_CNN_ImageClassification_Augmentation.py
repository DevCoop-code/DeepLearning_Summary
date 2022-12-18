# Data Augmentation이 적용된 CIFAR10 데이터 다운로드

train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10", 
                                train = True, 
                                download = True, 
                                transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10", 
                                train = False, 
                                transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)


# torchvision Module 내에 있는 transforms 함수를 이용시 Augmentation을 손쉽게 적용가능

# transforms.Compose() : 불러오는 이미지 데이터 전처리 및 Augmentation을 다양하게 적용할 때 이용하는 메서드
# transforms.RandomHorizontalFlip() : 해당 이미지를 50%의 확률로 좌우 반전
# transforms.Normalize : 표준편차를 의미, red, green, blue 순으로 표준편차 0.5씩 적용하는 것을 의미, Tensor 형태로 이미지 전환시 또 다른 정규화를 하는데 이때 평균과 표준편차가 필요
