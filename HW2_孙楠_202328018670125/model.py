# 0.0 MODULE
import torch
import os
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

# 0.1 data 
trans_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4), # random crop to 32*32 then pad 4
    transforms.RandomHorizontalFlip(0.5), # 50% HorizontalFlip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

trans_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=trans_valid)
testloader = DataLoader(testset, batch_size=256,shuffle=False)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse ', 'ship', 'truck ')

# 1.0 CLASS:Attention
class Attention(nn.Module):
    def __init__(self, dim=128, heads=8, dim_head = 64, dropout=0.1) -> None:
        super(Attention,self).__init__()
        self.heads = heads
        self.dim = dim
        self.dim_head =  dim_head
        self.scale = self.dim_head ** -0.5 

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        inner_dim = dim_head * heads
        project_out = not(self.heads == 1 and self.dim_head == dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self,x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3,dim=-1)       # [batch_size, seq_len, inner_dim * 3] --> [batch_size, seq_len, inner_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d)->b h n d', h=self.heads), qkv) # bs numinput heads dim
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) 


# 1.1 CLASS: ViT
class ViT(nn.Module):
    def __init__(self, num_classes=10, dim=512, depth=6, heads=8, mlp_dim=512, pool='cls', channels=3, dim_head=64,
                 dropout=0.2, emb_dropout=0.2):
        super().__init__()
        image_height = 32
        patch_height = 4
        image_width = 32
        patch_width = 4
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2)->b (h w) (p1 p2 c) ', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim), nn.LayerNorm(dim), )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Encoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

# 1.2 CLASS:Encoder
class Encoder(nn.Module):
    def __init__(self, dim=512, depth=12, heads=8, dim_head=64, mlp_dim=4096, dropout=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# 1.3 CLASS:FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
import time
from tqdm import tqdm
# 2.0 FUNC：train
def train(epoch, mode=1):
    
    print(f'\n total Epoch: {epoch}')
    model = ViT()
    device = 'cuda:1'
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    net = model.to(device)
    net.to(device)
    train_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    t = time.time()
    loss_all=[]
    acc_=[]


    # 尝试加载先前训练好的模型
    if os.path.exists('./model_ViT.pt'):
        state_dict = torch.load('./model_ViT.pt', map_location=device)
        model.load_state_dict(state_dict)
        print("\033[91m Loaded the pretrained model successfully \033[0m")

    if mode == 0:
        acc1=te(net, "cuda:1", criterion)
        print(f"only test: \033[92m acc: {acc1}\033[0m")
        return 0
    
    for e in range(epoch):
        net.train()
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=trans_train)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {e+1}/{epoch}")):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets) 
            loss.backward()
            
            # sparse_selection()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        loss_all.append(float(loss))
        acc1=te(net, device, criterion)
        acc_.append(float(acc1))
        print(f"epoch: {e+1}, \033[92m acc: {acc1}\033[0m, loss: {loss}")

        print(f"timecost: {time.time() - t}s")
    with open('loss.txt','w',encoding="utf-8") as f:
        f.write(str(loss_all))
    with open('acc.txt','w',encoding="utf-8") as f:
        f.write(str(acc_))
    torch.save(model.state_dict(),'./model_ViT.pt')

# 2.1 FUNC:te
def te(net, device, criterion):
    test_loss = 0
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _,predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total

train(300,mode=1)

# train(1,mode=0) #这是测试

# 4 min per epoch at MAC CPU bs=256
# 3 min per epoch at A10 GPU bs=256
# 0.5 min per epoch at A10 GPU bs=1024