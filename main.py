import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# generator
class block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class generator(nn.Module):
    def __init__(self, in_channels=3, feature=64):
        super().__init__()
        self.initial_down = nn.Sequential(nn.Conv2d(in_channels, feature, 4, 2, 1, padding_mode="reflect"),
                                          nn.LeakyReLU(0.2)
                                          )
        self.down1 = block(feature, feature * 2, down=True, act="relu", use_dropout=False)
        self.down2 = block(feature * 2, feature * 4, down=True, act="relu", use_dropout=False)
        self.down3 = block(feature * 4, feature * 8, down=True, act="relu", use_dropout=False)
        self.down4 = block(feature * 8, feature * 8, down=True, act="relu", use_dropout=False)
        self.down5 = block(feature * 8, feature * 8, down=True, act="relu", use_dropout=False)
        self.down6 = block(feature * 8, feature * 8, down=True, act="relu", use_dropout=False)
        self.bottleneck = nn.Sequential(nn.Conv2d(feature * 8, feature * 8, 4, 2, 1),
                                        nn.ReLU()
                                        )
        self.up1 = block(feature * 8, feature * 8, down=False, act="relu", use_dropout=True)
        self.up2 = block(feature * 8 * 2, feature * 8, down=False, act="relu", use_dropout=True)
        self.up3 = block(feature * 8 * 2, feature * 8, down=False, act="relu", use_dropout=True)
        self.up4 = block(feature * 8 * 2, feature * 8, down=False, act="relu", use_dropout=False)
        self.up5 = block(feature * 8 * 2, feature * 4, down=False, act="relu", use_dropout=False)
        self.up6 = block(feature * 4 * 2, feature * 2, down=False, act="relu", use_dropout=False)
        self.up7 = block(feature * 2 * 2, feature, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(feature * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


# Building Discriminator

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),

        )

    def forward(self, img_A, img_B):
        # Here we are concatenating the images on their channels
        img_input = torch.cat((img_A, img_B), 1)
        img_input = self.model(img_input)
        return img_input


model = generator(in_channels=3, feature=64)

data_dir = "datasets/maps"

data_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.CenterCrop((256, 512)),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder(root=data_dir, transform=data_transform)
dataset_val = datasets.ImageFolder(root=data_dir, transform=data_transform)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=24, shuffle=True, num_workers=2)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=24, shuffle=True, num_workers=2)


Normalization_Values = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def DeNormalize(tensor_of_image):
    return tensor_of_image * Normalization_Values[1][0] + Normalization_Values[0][0]


def save_images(image_tensor, num_images, folder,title,epoch):
    images = DeNormalize(image_tensor)
    images = images.detach().cpu()
    image_grid = make_grid(images[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    # plt.show()
    plt.savefig(folder + f'/image_{title}_{epoch}.png')  # 保存图像到指定文件夹



criterion = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

discriminator = Discriminator()


# discriminator loss
def disc_loss(inputs, targets, discriminator_opt):
    discriminator_opt.zero_grad()
    output = discriminator(inputs, targets)
    label = torch.ones(size=output.shape, dtype=torch.float, device=device)
    real_loss = criterion(output, label)

    gen_img = generator(inputs).detach()
    fake_output = discriminator(inputs, gen_img)
    fake_label = torch.zeros(size=fake_output.shape, dtype=torch.float, device=device)
    fake_loss = criterion(fake_output, fake_label)

    loss = (real_loss + fake_loss) / 2
    loss.backward()
    discriminator_opt.step()
    return loss


# generator loss
def gen_loss(inputs, targets, genrerator_opt, L1_lambda):
    genrerator_opt.zero_grad()
    gen_img = generator(inputs)
    fake_output = discriminator(inputs, gen_img)
    real_label = torch.ones(size=fake_output.shape, dtype=torch.float, device=device)

    loss = criterion(fake_output, real_label) + L1_lambda * torch.abs(gen_img - targets).sum()
    loss.backward()
    genrerator_opt.step()
    return loss, gen_img


L1_lambda = 100
NUM_EPOCHS = 20
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

discriminator = discriminator.to(device)
generator = generator(in_channels=3, feature=64).to(device)
discriminator_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
generator_opt = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

if __name__ == '__main__':
    for epoch in range(NUM_EPOCHS + 1):
        print(f"Training epoch {epoch + 1}")
        for images, _ in tqdm(dataloader_train):
            # Splitting the image

            inputs = images[:, :, :, :256].to(device)
            targets = images[:, :, :, 256:].to(device)

            # Train Discriminator

            Disc_Loss = disc_loss(inputs, targets, discriminator_opt)

            # Train Geneerator
            for i in range(2):
                Gen_Loss, generator_image = gen_loss(inputs, targets, generator_opt, L1_lambda)

        if (epoch % 5) == 0:
            save_images(inputs, 5,'./examples',"input",epoch)
            save_images(generator_image, 5,'./examples',"predict",epoch)
            save_images(targets, 5,'./examples',"target",epoch)
