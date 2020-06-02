import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

class ImgVisualization:
    # path_to_img_file or img_file (nifti1), path_to_annotations, list of strings
    def __init__(self, img_file, annotation_file="empty", pref_ornt=("L", "A", "S")):
        if isinstance(img_file, str):
            # load img regardless of orientation
            pre_img = nib.load(img_file)
        else:
            pre_img = img_file
        pre_img_data = pre_img.get_fdata()
        print(pre_img_data.shape)
        # Change orientation to desired orientation
        self.pref_ornt = pref_ornt
        if not nib.aff2axcodes(pre_img.affine) == self.pref_ornt:
            self.img = self.check_orientation(pre_img)
        else:
            self.img = pre_img

        if not annotation_file == "empty":
            #self.annotations = JsonToAnnotation(annotation_file, self.img).create_annotations()
            pass
        else:
            self.annotations="empty"

        # array.shape = [z, y, x]
        self.data = self.img.get_fdata().astype(np.float32)
        print(self.data.shape)
        self.size = self.data.shape

        self.layer = - ( - len(self.data[0]) // 2)
        self.orient_index = 0
        self.mouse_coord = [0, 0]
        self.max_mode = False

        plt.imshow(self.get_current_layer(), cmap="gray")

        self.fig = plt.gca()

        self.vmin, self.vmax = plt.gci().get_clim()
        self.v_center = (self.vmin + self.vmax) / 2
        self.v_width = 100

        self.im = plt.imshow(self.get_current_layer(), cmap="gray",
                             vmax=self.get_current_v()[0] + self.get_current_v()[1],
                             vmin=self.get_current_v()[0] - self.get_current_v()[0])

        self.cid_scroll = self.fig.figure.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_on_press = self.fig.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_on_motion = self.fig.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.figure.canvas.mpl_connect("key_press_event", self.on_key)

    def check_orientation(self, pre_img):
        # returns Nifti1Image with desired orientation and affine matrix
        img = pre_img
        orig_img = pre_img
        new_ornt = nib.orientations.axcodes2ornt(self.pref_ornt)
        affine = img.affine
        img_data = img.get_fdata().astype(int)
        orig_ornt = nib.io_orientation(affine)
        ornt_trans = nib.orientations.ornt_transform(orig_ornt, new_ornt)
        orig_shape = img_data.shape
        new_img_data = nib.orientations.apply_orientation(img_data, ornt_trans)
        aff_trans = nib.orientations.inv_ornt_aff(ornt_trans, orig_shape)
        self.trans_mat = aff_trans
        print(aff_trans)
        new_affine = np.dot(affine, aff_trans)
        img = nib.Nifti1Image(new_img_data, new_affine, img.header)
        return img

    def plot(self):
        self.im.set_data(self.get_current_layer())
        self.im.set_clim(vmax=self.get_current_v()[0] + self.get_current_v()[1],
                         vmin=self.get_current_v()[0] - self.get_current_v()[0])
        if not self.annotations == "empty":
            for point in self.annotations[0]["points"]:
                ax = plt.gca()
                if self.orient_index == 0:
                    if point["coords"][0] == self.layer:
                        ax.add_artist(plt.Circle((point["coords"][1], point["coords"][2]),
                                                    radius=point["size"] / 2,
                                                    color=point["color"]))
                elif self.orient_index == 1:
                    if point["coords"][1] == self.layer:
                        ax.add_artist(plt.Circle((point["coords"][0], point["coords"][2]),
                                                    radius=point["size"] / 2,
                                                    color=point["color"]))
                elif self.orient_index == 2:
                    if point["coords"][2] == self.layer:
                        ax.add_artist(plt.Circle((point["coords"][0], point["coords"][1]),
                                                 radius=point["size"] / 2,
                                                 color=point["color"]))
                else: continue
        else:
            pass

    def get_current_layer(self):
        if self.get_current_max_mode() and 2 < self.layer < len(self.data) + 1:
            if self.orient_index == 0:
                return np.max(self.data[(self.layer - 2):(self.layer + 2), :, :], axis=self.orient_index)
            elif self.orient_index == 1:
                return np.max(self.data[:, (self.layer - 2):(self.layer + 2), :], axis=self.orient_index)
            elif self.orient_index == 2:
                return np.max(self.data[:, :, (self.layer - 2):(self.layer + 2)], axis=self.orient_index)
        else:
            if self.orient_index == 0:
                return self.data[self.layer, :, :]
            elif self.orient_index == 1:
                return self.data[:, self.layer, :]
            elif self.orient_index == 2:
                return self.data[:, :, self.layer]

    def get_current_v(self):
        return [self.v_center, self.v_width]

    def get_current_max_mode(self):
        return self.max_mode

    def on_key(self, event):
        if event.key == "t":
            print(event.key)
            if self.orient_index < 2:
                self.orient_index += 1
            else:
                self.orient_index = 0

            ax = plt.gca()
            if self.orient_index == 0:
                ax.set(xlim=(0, self.size[2]), ylim=(0, self.size[2]))
            elif self.orient_index == 1:
                ax.set(xlim=(0, self.size[2]), ylim=(0, self.size[0]))
            elif self.orient_index == 2:
                ax.set(xlim=(0, self.size[0]), ylim=(0, self.size[1]))


            self.plot()
        elif event.key == "m":
            if not self.max_mode:
                self.max_mode = True
            else:
                self.max_mode = False
            print("max_mode: " + str(self.get_current_max_mode()))
        plt.draw()

    def on_scroll(self, event):
        if event.button == "up":
            if event.key == "shift":
                self.layer -= 5
                self.fig.set_title(str(self.layer))
            else:
                self.layer -= 1
                self.fig.set_title(str(self.layer))
            self.plot()
        elif event.button == "down":
            if event.key == "shift":
                self.layer += 5
                self.fig.set_title(str(self.layer))
            else:
                self.layer += 1
                self.fig.set_title(str(self.layer))
            self.plot()
        plt.draw()

    def on_press(self, event):
        if event.inaxes and event.button == 1 and event.key == "shift":
            self.mouse_coord = [event.xdata, event.ydata]

    def on_motion(self, event):
        if event.inaxes and event.key == "shift":
            d_x = event.xdata - self.mouse_coord[0]
            d_y = event.ydata - self.mouse_coord[1]
            self.v_center = ((self.vmax + self.vmin) / 2) + d_x
            self.v_width = 100 + d_y
        self.plot()
        plt.draw()

    def disconnect(self):
        self.fig.figure.canvas.mpl_disconnect(self.cid_scroll)
        self.fig.figure.canvas.mpl_disconnect(self.cid_on_press)
        self.fig.figure.canvas.mpl_disconnect(self.cid_on_motion)
        self.fig.figure.canvas.mpl_disconnect(self.cid_key)