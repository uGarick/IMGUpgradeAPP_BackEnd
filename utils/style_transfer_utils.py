import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import copy

# Определяем устройство (GPU, если доступно)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(image_name, imsize=512):
    """
    Загружает и преобразует изображение в тензор.
    """
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)


def imsave(tensor, path):
    """
    Сохраняет тензор как изображение.
    """
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(path)


def gram_matrix(input_tensor):
    """
    Вычисляет матрицу Грама для признаков.
    """
    batch_size, f, h, w = input_tensor.size()
    features = input_tensor.view(batch_size * f, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * f * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


def neural_style_transfer(content_image_path, style_image_path, output_path,
                          num_steps=300, style_weight=1000000, content_weight=1):
    """
    Применяет нейронный перенос стиля:
    - content_image_path: путь к изображению с контентом
    - style_image_path: путь к изображению со стилем
    - output_path: путь для сохранения результирующего изображения
    """
    content_img = image_loader(content_image_path)
    style_img = image_loader(style_image_path)

    both_size = content_img.size()
    style_img = F.interpolate(style_img, size=(both_size[2], both_size[3]), mode='bilinear', align_corners=False)

    if style_img.size()[1] != content_img.size()[1]:
        style_img = style_img[:, :content_img.size()[1], :, :]

    print(content_img.size(), style_img.size())

    assert content_img.size() == style_img.size(), "Размеры изображений должны совпадать"

    # Загружаем предобученную модель VGG19
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Модуль нормализации
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            return (img - self.mean) / self.std

    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            name = f"layer_{i}"

        model.add_module(name, layer)

        if name in content_layers_default:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)


        if name in style_layers_default:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Обрезаем модель после последнего слоя потерь
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    # Инициализируем оптимизируемым изображением как копию контентного
    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Шаг {run[0]}: стиль {style_score.item():.4f} контент {content_score.item():.4f}")
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    imsave(input_img, output_path)
    print(f"Нейронная стилизация завершена, результат сохранен как {output_path}")
