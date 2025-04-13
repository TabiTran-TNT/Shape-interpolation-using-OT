from utils import *
from model import *
import torch_directml
from train_UOT import train
from generate_data import *
from visualize import plot3d, axplot3d
import polyscope as ps

class TempArgs:
    def __init__(self):
        # training parameters
        self.num_epochs = 32
        self.batch_size = 1024
        #self.mass_weight = mass_weight_
        #self.energy_weight = energy_weight_  # IMPORTANT!, a good energy weights balance well the energy value with data fedelity term(MSE term)
        self.learning_rate = 1e-3
        self.use_gpu = True
        self.device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
        # self.device = torch_directml.device() if torch_directml.device() else \
        # torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
        self.N_Selected = 8
        self.T = 19  # important, it determine how many intermidiate images we have
        self._tol = float(1e-2)  # to threshold the images, 0 will cause singularity. 1e-3 works well
        self.tau = 10000
        #self.boundary = boundary_  # 'periodic', 'dirichlet', 'neumann'
        self.bceloss = True
        self.imgcuroption = 'mid'
        # training dataset
        self.m = 32  # important, large image size cause slow computation of the subproblem #when choose then size of images, it'd have to be from 18,24,32,40,48 (the multiple of 8 since the NN has 4 pooling layer)
        self.n = 32  # same as m
        self.channel = 1
        self.shuffle = True
        self._pad_value = 1
        self._mass_standard = 400
        self.obstacleoption = 'default'
        self.first_and_last = None
        self.flip = True
        # global variables
        self.v_dim = 2 * self.m * self.n + self.m + self.n
        self.spdiv = None  # this is the sparse divergence operator, since creating it every time requires large RAM when m n are large, so it's easier to load it from local
        self.mask = None  # this is due to boundary condition of divergence operator, For now, the momentum on the boundary are eliminated. In the future, we consider block some area
        self.obstacle = None
        self.steps = (torch.arange(self.T + 1, device=self.device) / self.T).unsqueeze(-1)
        self.pathvariables = None
        # verbosity options
        self.regu_verbosity = True
        self.regu_path_verbosity = True
        self.imshow_gap = 400
        self.save_gap = 10

    @property
    def mass_standard(self):
        return self._mass_standard

    @mass_standard.setter
    def mass_standard(self, value):
        print('args.mass_standard changed from ', self._mass_standard, ' to new value', value)
        self._mass_standard = float(value)

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        print('args.tol changed from ', self._tol, ' to new value', value)
        self._tol = float(value)

    @property
    def diff_tol(self):
        return self._diff_tol

    @diff_tol.setter
    def diff_tol(self, value):
        print('args.diff_tol changed from ', self._diff_tol, 'to new value', value)
        self._diff_tol = float(value)

    @property
    def pad_value(self):
        return self._pad_value

    @pad_value.setter
    def pad_value(self, value):
        print('args.pad_value changed from', self._pad_value, ' to new value', value)
        self._pad_value = float(value)

def main_OT(mass_weight_, energy_weight_, boundary_ ):
  #training parameter settings and some global variables are also saved here
  #there are many many parameters which is not used for now, the important ones are commented
  class Args:
      def __init__(self):
          #training parameters
          self.num_epochs = 10002
          self.batch_size = 1024
          self.mass_weight = mass_weight_
          self.energy_weight = energy_weight_ #IMPORTANT!, a good energy weights balance well the energy value with data fedelity term(MSE term)
          self.learning_rate = 1e-3
          self.use_gpu = True
          self.device = torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
          # self.device = torch_directml.device() if torch_directml.device() else \
            # torch.device("cuda:0" if self.use_gpu and torch.cuda.is_available() else "cpu")
          self.N_Selected = 8
          self.T = 19  #important, it determine how many intermidiate images we have
          self._tol = float(1e-2) #to threshold the images, 0 will cause singularity. 1e-3 works well
          self.tau  = 10000
          self.boundary = boundary_ #'periodic', 'dirichlet', 'neumann'
          self.bceloss = True
          self.imgcuroption = 'mid'
          #training dataset
          self.m = 32  #important, large image size cause slow computation of the subproblem #when choose then size of images, it'd have to be from 18,24,32,40,48 (the multiple of 8 since the NN has 4 pooling layer)
          self.n = 32  #same as m
          self.channel = 1
          self.shuffle = True
          self._pad_value = 1
          self._mass_standard = 400
          self.obstacleoption = 'default'
          self.first_and_last = None
          self.flip = True
          #global variables
          self.v_dim = 2*self.m*self.n+self.m+self.n
          self.spdiv = None #this is the sparse divergence operator, since creating it every time requires large RAM when m n are large, so it's easier to load it from local
          self.mask = None # this is due to boundary condition of divergence operator, For now, the momentum on the boundary are eliminated. In the future, we consider block some area
          self.obstacle = None
          self.steps = (torch.arange(self.T+1, device=self.device) / self.T).unsqueeze(-1)
          self.pathvariables = None
          #verbosity options
          self.regu_verbosity = True
          self.regu_path_verbosity = True
          self.imshow_gap = 40000
          self.save_gap = 500

      @property
      def mass_standard(self):
          return self._mass_standard

      @mass_standard.setter
      def mass_standard(self, value):
          print('args.mass_standard changed from ',  self._mass_standard, ' to new value', value)
          self._mass_standard = float(value)

      @property
      def tol(self):
          return self._tol

      @tol.setter
      def tol(self, value):
          print('args.tol changed from ', self._tol, ' to new value', value)
          self._tol = float(value)

      @property
      def diff_tol(self):
          return self._diff_tol

      @diff_tol.setter
      def diff_tol(self, value):
          print('args.diff_tol changed from ', self._diff_tol  ,'to new value', value)
          self._diff_tol = float(value)

      @property
      def pad_value(self):
          return self._pad_value

      @pad_value.setter
      def pad_value(self, value):
          print('args.pad_value changed from', self._pad_value ,' to new value', value)
          self._pad_value = float(value)

  args = Args()
  print(args.device)

  #%%script false --no-raise-error

  # step 1: load training data, currently only two images
  # train_dataloader = loadimages(args, 32, 32, imgs=[file_path + 'channel.png',file_path + 'crowd.png'],flip_= args.flip,gray_=True)
  # train_dataloader, test_dataloader = loadMNIST(args, True, 10000)
  shape1 = np.zeros((32, 32, 32))
  shape2 = np.zeros((32, 32, 32))

  pc1 = get_donut_point_cloud(120).T
  pc2 = get_duck_point_cloud(120).T

  print(pc1)

  # shape 1
  max_x = np.max(pc1[:, 0])
  min_x = np.min(pc1[:, 0])
  max_y = np.max(pc1[:, 1])
  min_y = np.min(pc1[:, 1])
  max_z = np.max(pc1[:, 2])
  min_z = np.min(pc1[:, 2])
  min_val = min_x
  max_val = max_x
  if abs(max_val - min_val) < abs(max_y - min_y):
      min_val = min_y
      max_val = max_y

  if abs(max_val - min_val) < abs(max_z - min_z):
      min_val = min_z
      max_val = max_z

  for point in pc1:
      x = int(np.floor(((point[0] - min_x) / (max_val - min_val)) / (1 / 32)))
      y = int(np.floor(((point[1] - min_y) / (max_val - min_val)) / (1 / 32)))
      z = int(np.floor(((point[2] - min_z) / (max_val - min_val)) / (1 / 32)))
      if x > 31:
          x = 31
      if y > 31:
          y = 31
      if z > 31:
          z = 31

      print(type(x))
      shape1[x, y, z] = 1.0

  # shape 2
  max_x = np.max(pc2[:, 0])
  min_x = np.min(pc2[:, 0])
  max_y = np.max(pc2[:, 1])
  min_y = np.min(pc2[:, 1])
  max_z = np.max(pc2[:, 2])
  min_z = np.min(pc2[:, 2])
  min_val = min_x
  max_val = max_x
  if abs(max_val - min_val) < abs(max_y - min_y):
      min_val = min_y
      max_val = max_y

  if abs(max_val - min_val) < abs(max_z - min_z):
      min_val = min_z
      max_val = max_z

  for point in pc2:
      x = int(np.floor(((point[0] - min_x) / (max_val - min_val)) / (1 / 31)))
      y = int(np.floor(((point[1] - min_y) / (max_val - min_val)) / (1 / 31)))
      z = int(np.floor(((point[2] - min_z) / (max_val - min_val)) / (1 / 31)))
      if x > 31:
          x = 31
      if y > 31:
          y = 31
      if z > 31:
          z = 31
      shape2[x, y, z] = 1.0

  shape1_lst = []
  for i in range(32):
      for j in range(32):
          for k in range(32):
              if shape2[i, j, k] > 0.0:
                  shape1_lst.append([i, j, k])
  shape1_lst = np.array(shape1_lst)

  plot3d(shape1_lst.T, axis_on=True)
  # plt.show()

  print(shape1.shape)
  train_dataloader = load3dshapes(args, 32, 32, [shape1, shape2])

  # step 2: create or load divergence operator and mask
  if args.boundary == 'periodic':
    create_div_periodicboundary(args)
    import_obstacle_periodicboundary(args)
  else:
    if args.boundary == 'neumann':
      create_div_Neumann(args)
    else:
      create_div_Dirichlet(args)

    import_obstacle(args)

  #step 3: create the NN
  my_autoencoder = Autoencoder_s_3d().to(args.device)
  from torchsummary import summary
  summary( my_autoencoder, (args.channel, args.m,args.n),1)

  losses = {
        "train_loss_avg":[],
        "mseterm":[],
        "massterm":[],
        "pathenergy":[]
    }
  optimizer = torch.optim.Adam(params=my_autoencoder.parameters(), lr=args.learning_rate)


  # trainer = TrainOT()
  # trainer = TrainUOT()
  # trainer = TrainUOT2()
  # trainer = TrainRecononly()
  train(args, train_dataloader, my_autoencoder,optimizer, losses, num_epochs= args.num_epochs)


  #fig = plt.figure(10,10)
  # plt.plot(losses["train_loss_avg"])
  # plt.xlabel('Epochs')
  # plt.ylabel('Reconstruction error')
  # plt.show()
  #
  # visualize_path(args, train_dataloader, my_autoencoder,flip_= False, T=9)
  return train_dataloader, my_autoencoder
  return train_dataloader, test_dataloader, my_autoencoder

def calculate_accuracy(output, target):
    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Get the predicted class by finding the index with the highest probability
    _, predicted = torch.max(probabilities, 1)

    # Compare predicted classes with the true labels
    correct = (predicted == target).sum().item()

    # Calculate accuracy
    accuracy = correct / target.size(0)
    return accuracy

# train_dataloader, autoencoder = main_OT(1e-5, 0, 'neumann')
autoencoder = Autoencoder_s_3d()
autoencoder.load_state_dict(torch.load("models/model_UOT_3D_6499.pth", weights_only=True))
autoencoder.eval()

shape1 = np.zeros((32, 32, 32))
shape2 = np.zeros((32, 32, 32))

pc1 = get_donut_point_cloud(120).T
pc2 = get_duck_point_cloud(120).T

print(pc1)

# shape 1
max_x = np.max(pc1[:, 0])
min_x = np.min(pc1[:, 0])
max_y = np.max(pc1[:, 1])
min_y = np.min(pc1[:, 1])
max_z = np.max(pc1[:, 2])
min_z = np.min(pc1[:, 2])
min_val = min_x
max_val = max_x
if abs(max_val - min_val) < abs(max_y - min_y):
  min_val = min_y
  max_val = max_y

if abs(max_val - min_val) < abs(max_z - min_z):
  min_val = min_z
  max_val = max_z

for point in pc1:
  x = int(np.floor(((point[0] - min_x) / (max_val - min_val)) / (1 / 32)))
  y = int(np.floor(((point[1] - min_y) / (max_val - min_val)) / (1 / 32)))
  z = int(np.floor(((point[2] - min_z) / (max_val - min_val)) / (1 / 32)))
  if x > 31:
      x = 31
  if y > 31:
      y = 31
  if z > 31:
      z = 31

  print(type(x))
  shape1[x, y, z] = 1.0

# shape 2
max_x = np.max(pc2[:, 0])
min_x = np.min(pc2[:, 0])
max_y = np.max(pc2[:, 1])
min_y = np.min(pc2[:, 1])
max_z = np.max(pc2[:, 2])
min_z = np.min(pc2[:, 2])
min_val = min_x
max_val = max_x
if abs(max_val - min_val) < abs(max_y - min_y):
  min_val = min_y
  max_val = max_y

if abs(max_val - min_val) < abs(max_z - min_z):
  min_val = min_z
  max_val = max_z

for point in pc2:
  x = int(np.floor(((point[0] - min_x) / (max_val - min_val)) / (1 / 31)))
  y = int(np.floor(((point[1] - min_y) / (max_val - min_val)) / (1 / 31)))
  z = int(np.floor(((point[2] - min_z) / (max_val - min_val)) / (1 / 31)))
  if x > 31:
      x = 31
  if y > 31:
      y = 31
  if z > 31:
      z = 31
  shape2[x, y, z] = 1.0
args = TempArgs()
train_dataloader = load3dshapes(args, 32, 32, [shape1, shape2])

# fig, axes = plt.subplots(1, 2, figsize=(45, 6 * 1.4), subplot_kw={'projection': '3d'})
# fig.subplots_adjust(hspace=0.05)
# fig.subplots_adjust(wspace=0.05)
# print(list(enumerate(axes.flat)))
image_list = next(iter(train_dataloader))[0]
# for i, ax in enumerate(axes.flat):
#     shape = np.squeeze(image_list[i].detach().cpu().numpy())
#     shape_lst = []
#     for i1 in range(32):
#         for j in range(32):
#             for k in range(32):
#                 if shape[i1, j, k] > 0.0:
#                     shape_lst.append([i1, j, k])
#     shape_lst = np.array(shape_lst)
    # axplot3d(ax, shape_lst.T, axis_on=True)

    # plot3d(np.squeeze(image_list[i].detach().cpu().numpy()))
    # ax.imshow(np.squeeze(image_list[i].detach().cpu().numpy()), 'gray')
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
# plt.grid(False)
# plt.show()
# fig.savefig("mnist-samples.png", bbox_inches='tight')
encode1 = autoencoder.encoder(image_list[0].to(args.device))
encode2 = autoencoder.encoder(image_list[1].to(args.device))

encode3 = encode1 + 0.2 * (encode2 - encode1)
interpolation = autoencoder.decoder(torch.unsqueeze(encode3, 0))
shape = np.squeeze(interpolation.detach().cpu().numpy())
shape_lst = []
color_lst = []
for i1 in range(32):
    for j in range(32):
        for k in range(32):
            if shape[i1, j, k] > 0.03:
                shape_lst.append([i1, j, k])
                color_lst.append(shape[i1, j, k])

shape_lst = np.array(shape_lst)
color_lst = np.array(color_lst)
# import polyscope as ps
#
# # Initialize polyscope
# ps.init()
#
# ### Register a point cloud
# # `my_points` is a Nx3 numpy array
# ps_cloud = ps.register_point_cloud("my points", shape_lst)
# ps_cloud.add_color_quantity("rand colors", color_lst)
# ps_cloud.show()

# plot3d(shape_lst.T, axis_on=True, c=color_lst)
# plt.grid(False)
# plt.show()

# Image RGB -> VAE -> interpolation
# n * m * 32

# Point cloud => 32 * 32 * 32
# 32 * 32 * 32 unit cube [0, 1] x [0, 1] x [0, 1]
#

# visualize_path(args, test_dataloader, autoencoder,flip_= False, T=8)

def visualize_result():
    T = 6
    flip_ = False
    steps = (torch.arange(T + 1, device=args.device) / T).unsqueeze(-1) if T is not None else args.steps

    shape_lst = []
    with torch.no_grad():
        for image_batch in train_dataloader:

            from torchvision import datasets, transforms

            # Define a transform to convert the data to tensor
            transform = transforms.ToTensor()

            # Load the MNIST training dataset
            # train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

            # image = transforms.ToPILImage()(image_batch[0][0])  # Save the image as a PNG file
            # image.save('compare-1.png')
            # image = transforms.ToPILImage()(image_batch[0][1])  # Save the image as a PNG file
            # image.save('compare-2.png')

            # image_batch = train_data
            image_batch = image_batch[0].to(args.device)
            image_batch_code = autoencoder.encoder(image_batch)
            image_batch_recon = autoencoder(image_batch)

            # num_b = min(args.N_Selected, image_batch.shape[0]) - 1
            num_b = 1
            image_path = []
            for i in range(num_b):

                image_path.append(image_batch[i].unsqueeze(0))  # default type is torch.float32
                for step in steps[0:T]:
                    # print(step)
                    code_sequence_i = image_batch_code[i] + step * (image_batch_code[i + 1] - image_batch_code[i])
                    image_interpolated_i = autoencoder.decoder(torch.unsqueeze(code_sequence_i, 0))

                    image_path.append(image_interpolated_i.clone())
                image_path.append(image_batch[i + 1].unsqueeze(0))

            # image_path.append(image_batch[num_b].unsqueeze(0))
            image_path = torch.cat(image_path)
            if flip_:
                image_path = 1 - image_path
            fig, axes = plt.subplots(num_b, T + 2, figsize=(5 * (T + 2), num_b * 1.4), subplot_kw={'projection': '3d'})
            # fig.subplots_adjust(hspace=0)
            fig.subplots_adjust(wspace=0.05)
            print(list(enumerate(axes.flat)))
            print(image_path[1])
            for i, ax in enumerate(axes.flat):
                print(image_path[i].shape)
                shape = np.squeeze(image_path[i].detach().cpu().numpy())
                shape_ = []
                for i1 in range(32):
                    for j in range(32):
                        for k in range(32):
                            if shape[i1, j, k] > 0.03:
                                shape_.append([i1, j, k])
                shape_ = np.array(shape_)
                shape_lst.append(shape_)
                # axplot3d(ax, shape_lst.T, axis_on=True)
                # ax.imshow(np.squeeze(image_path[i].detach().cpu().numpy(), 0), 'gray')
                #
                # ax.grid(False)
                # ax.set_xticks([])
                # ax.set_yticks([])
                #
                # if i % (T + 2) == 0 or i % (T + 2) == (T + 2) - 1:
                #     frame_color = 'red'
                #     ax.spines['top'].set_color(frame_color)
                #     ax.spines['bottom'].set_color(frame_color)
                #     ax.spines['left'].set_color(frame_color)
                #     ax.spines['right'].set_color(frame_color)
                #
                #     frame_width = 3  # Increase the width
                #     ax.spines['top'].set_linewidth(frame_width)
                #     ax.spines['bottom'].set_linewidth(frame_width)
                #     ax.spines['left'].set_linewidth(frame_width)
                #     ax.spines['right'].set_linewidth(frame_width)
                # elif i % (T + 2) == 1 or i % (T + 2) == (T + 2) - 2:
                #     frame_color = 'blue'
                #     ax.spines['top'].set_color(frame_color)
                #     ax.spines['bottom'].set_color(frame_color)
                #     ax.spines['left'].set_color(frame_color)
                #     ax.spines['right'].set_color(frame_color)
                #
                #     frame_width = 3  # Increase the width
                #     ax.spines['top'].set_linewidth(frame_width)
                #     ax.spines['bottom'].set_linewidth(frame_width)
                #     ax.spines['left'].set_linewidth(frame_width)
                #     ax.spines['right'].set_linewidth(frame_width)

            # plt.grid(False)
            # plt.show()
            # fig.savefig("main-result-1.png", bbox_inches='tight')
            # break
            # make_grid_show(image_path, pad_value=args.pad_value)

    print(shape_lst)

    for i in range(0, len(shape_lst)):
        shape = shape_lst[i][:, [1, 0, 2]].copy()
        shape[:, 2] = -shape[:, 2]
        shape_lst[i] = shape

    color_lst = []
    for i in range(0, len(shape_lst)):
        color_lst.append(np.linalg.norm(shape_lst[i], axis=1) / 100)
        shape_lst[i][:, 0] += 40 * i
        shape_lst[i][:, 2] -= 20 * i
        # shape_lst[i] = rotate_y(shape_lst[i], -17)
        # shape_lst[i] = rotate_x(shape_lst[i], -15)

    result_shape = np.vstack(shape_lst)
    result_color = np.hstack(color_lst)
    print(result_color.shape)

    ps.init()
    ps.set_ground_plane_mode("none")

    ps.register_point_cloud("pd1", result_shape, point_render_mode="quad", radius=0.0015)
    ps.get_point_cloud("pd1").add_scalar_quantity("cl1", result_color)
    # ps.register_point_cloud("pd1", point_cloud_2)
    ps.show()

def visualize_compare():
    T = 9
    flip_ = False
    steps = (torch.arange(T + 1, device=args.device) / T).unsqueeze(-1) if T is not None else args.steps

    with torch.no_grad():
        for image_batch in train_dataloader:
            print(image_batch[0].shape)
            from torchvision import transforms
            from PIL import Image

            # Define a transform to convert the image to a tensor
            transform = transforms.ToTensor()

            # Apply the transform to the image
            image_batch = []
            image = Image.open("compare-1.png")
            image_tensor = transform(image)
            image_batch.append(image_tensor)
            image = Image.open("compare-2.png")
            image_tensor = transform(image)
            image_batch.append(image_tensor)
            image_batch = torch.stack(image_batch).to(args.device)
            print(image_batch.shape)

            image_batch_code = autoencoder.encoder(image_batch)
            image_batch_recon = autoencoder(image_batch)

            # num_b = min(args.N_Selected, image_batch.shape[0]) - 1
            num_b = 1
            image_path = []
            for i in range(num_b):

                image_path.append(image_batch[i].unsqueeze(0))  # default type is torch.float32
                for step in steps[0:T]:
                    # print(step)
                    code_sequence_i = image_batch_code[i] + step * (image_batch_code[i + 1] - image_batch_code[i])
                    image_interpolated_i = autoencoder.decoder(torch.unsqueeze(code_sequence_i, 0))

                    image_path.append(image_interpolated_i.clone())
                image_path.append(image_batch[i + 1].unsqueeze(0))

            # image_path.append(image_batch[num_b].unsqueeze(0))
            image_path = torch.cat(image_path)
            if flip_:
                image_path = 1 - image_path
            fig, axes = plt.subplots(num_b, T + 2, figsize=(5 * (T + 2), num_b * 1.4))
            # fig.subplots_adjust(hspace=0)
            fig.subplots_adjust(wspace=0.05)
            print(list(enumerate(axes.flat)))
            for i, ax in enumerate(axes.flat):
                ax.imshow(np.squeeze(image_path[i].detach().cpu().numpy(), 0), 'gray')

                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])

                if i % (T + 2) == 0 or i % (T + 2) == (T + 2) - 1:
                    frame_color = 'red'
                    ax.spines['top'].set_color(frame_color)
                    ax.spines['bottom'].set_color(frame_color)
                    ax.spines['left'].set_color(frame_color)
                    ax.spines['right'].set_color(frame_color)

                    frame_width = 3  # Increase the width
                    ax.spines['top'].set_linewidth(frame_width)
                    ax.spines['bottom'].set_linewidth(frame_width)
                    ax.spines['left'].set_linewidth(frame_width)
                    ax.spines['right'].set_linewidth(frame_width)
                elif i % (T + 2) == 1 or i % (T + 2) == (T + 2) - 2:
                    frame_color = 'blue'
                    ax.spines['top'].set_color(frame_color)
                    ax.spines['bottom'].set_color(frame_color)
                    ax.spines['left'].set_color(frame_color)
                    ax.spines['right'].set_color(frame_color)

                    frame_width = 3  # Increase the width
                    ax.spines['top'].set_linewidth(frame_width)
                    ax.spines['bottom'].set_linewidth(frame_width)
                    ax.spines['left'].set_linewidth(frame_width)
                    ax.spines['right'].set_linewidth(frame_width)

            plt.grid(False)
            plt.show()
            fig.savefig("test-result-1.png", bbox_inches='tight')
            break
            # make_grid_show(image_path, pad_value=args.pad_value)

visualize_result()
# visualize_compare()
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
#
# # Define a simple neural network
# class SimpleNN(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(SimpleNN, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc0 = nn.Linear(256, 128)
#         self.fc1 = nn.Linear(128, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.1)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.relu(self.fc0(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         x = self.softmax(x)
#         return x
#
# # # Initialize the model, loss function, and optimizer
# # model = SimpleNN(256, 10).to("cpu")
# # criterion = nn.CrossEntropyLoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)
# #
# # # Training loop
# # num_epochs = 200
# # for epoch in range(num_epochs):
# #     model.train()
# #     for features, labels in train_dataloader:
# #         features = features.to("cpu")
# #         inputs = autoencoder.encoder(features)
# #         optimizer.zero_grad()
# #         outputs = model(inputs)
# #         loss = criterion(outputs, F.one_hot(labels, num_classes=10).double().to("cpu"))
# #         loss.backward()
# #         optimizer.step()
# #         print(calculate_accuracy(outputs, labels))
# #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#
#
# model = SimpleNN(256, 10).to("cpu")
# model.load_state_dict(torch.load("classifier.pth", weights_only=True))
# model.eval()
#
#
# for features, labels in test_dataloader:
#     features = features.to("cpu")
#     inputs = autoencoder.encoder(features)
#     outputs = model(inputs)
#     print(calculate_accuracy(outputs, labels))
