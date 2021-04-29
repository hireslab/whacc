import h5py
import copy
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
# import time


# from whacc import utils

# H5_file_name = '/Users/phil/Dropbox/Autocurator/testing_data/MP4s/AH0667x170317_JON/AH0667x170317-.h5'
# label_key = 'labels'

def touch_gui(H5_file_name, label_read_key, label_write_key=None):
    """

    Parameters
    ----------
    H5_file_name :
        
    label_read_key :
        
    label_write_key :
         (Default value = None)

    Returns
    -------

    """
    if label_write_key is None:
        label_write_key = label_read_key
    global im_ind
    global h5
    global img
    global LABELS



    with h5py.File(H5_file_name, 'r+') as h5:
        def move(add_to):
            """

            Parameters
            ----------
            add_to :
                

            Returns
            -------

            """
            global im_ind
            global h5
            global img
            global LABELS
            ###
            im_ind = im_ind + add_to
            xx = 3  # how many pictures on each side of the center picture
            img = np.zeros(np.asarray(h5['images'][0].shape) + np.asarray([10, 2, 0])).astype('int16')
            img = np.repeat(img, xx * 2 + 1, axis=1)
            img = np.concatenate((img, np.expand_dims(img[:, 0, :], axis=[1])), axis=1)
            img = img[:, 1:, :]
            W = h5['images'][0].shape[1]
            for i, k in enumerate(index_out(h5['images'], im_ind - xx, im_ind + xx + 1)):
                b = h5['images'][k]
                l = LABELS[k]
                if l == -1:
                    pass
                elif l > 0.5:
                    img[:, i * (W + 2):(i + 1) * (W + 2), :] = np.expand_dims(np.asarray([0, 255, 0]), axis=[0, 1])
                elif l < 0.5:
                    img[:, i * (W + 2):(i + 1) * (W + 2), :] = np.expand_dims(np.asarray([255, 0, 0]), axis=[0, 1])
                img[1:1 + b.shape[0], 2 * i + 1 + b.shape[1] * i: 2 * i + 1 + b.shape[1] * (i + 1), :] = b
            a = np.ones([10, img.shape[1], 3]) * 255
            x2 = img.shape[1] / xx
            a[:, int(img.shape[1] / (xx * 2 + 1) * xx):int(img.shape[1] / (xx * 2 + 1) * (xx + 1)), :] = 0
            img = np.concatenate((img, a), axis=0)
            im = Image.fromarray(img.astype('uint8'))
            im = im.resize(np.flip(np.asarray(img.shape[:2]) * 3), Image.ANTIALIAS)
            image_tk = ImageTk.PhotoImage(image=im)

            label_1.configure(image=image_tk)
            label_1.image = image_tk

            ind2print = copy.deepcopy(im_ind) + 1
            if im_ind < 0:
                ind2print = len(LABELS) - (im_ind * -1) + 1
            text.configure(text=str(ind2print) + ' of ' + str(len(LABELS)))
            text.text = str(ind2print) + ' of ' + str(len(LABELS))

        def index_out(a, start, stop):
            """

            Parameters
            ----------
            a :
                
            start :
                
            stop :
                

            Returns
            -------

            """
            if stop <= start:
                stop += len(a)
            return np.arange(start, stop) % len(a)

        def move_right(event=None):
            """

            Parameters
            ----------
            event :
                 (Default value = None)

            Returns
            -------

            """
            global im_ind
            move(1)

        def move_left(event=None):
            """

            Parameters
            ----------
            event :
                 (Default value = None)

            Returns
            -------

            """
            global im_ind
            move(-1)

        def switch_label(event=None):
            """

            Parameters
            ----------
            event :
                 (Default value = None)

            Returns
            -------

            """
            global LABELS
            global im_ind
            if LABELS[im_ind] == -1:
                pass
            elif LABELS[im_ind] > .5:
                LABELS[im_ind] = 0
            elif LABELS[im_ind] < .5:
                LABELS[im_ind] = 1
            move(0)

        def make_1(event=None):
            """

            Parameters
            ----------
            event :
                 (Default value = None)

            Returns
            -------

            """
            global LABELS
            global im_i
            LABELS[im_ind] = 1
            move(0)

        def make_0(event=None):
            """

            Parameters
            ----------
            event :
                 (Default value = None)

            Returns
            -------

            """
            global LABELS
            global im_i
            LABELS[im_ind] = 0
            move(0)

        def make_neg1(event=None):
            """

            Parameters
            ----------
            event :
                 (Default value = None)

            Returns
            -------

            """
            global LABELS
            global im_i
            LABELS[im_ind] = -1
            move(0)

        def save_foo(event=None):
            """

            Parameters
            ----------
            event :
                 (Default value = None)

            Returns
            -------

            """
            global LABELS
            neg_ones = np.where(LABELS == -1)
            LABELS = (LABELS>.5)*1
            for neg_ind in neg_ones:
                LABELS[neg_ind] = -1
            try:
                del h5[label_write_key]
                # time.sleep(10)  # give time to process the deleted file... maybe???
                h5.create_dataset(label_write_key, data=np.float64(LABELS))
            except:
                h5.create_dataset(label_write_key, data=np.float64(LABELS))

        root = Tk()
        im = None
        LABELS = copy.deepcopy(h5[label_read_key][:])
        c = ttk.Frame(root, padding=(5, 20, 6, 0))
        c.grid(column=0, row=0, sticky=(N, W, E, S))
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        im_ind = -1
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(h5['images'][im_ind]))

        set_1 = ttk.Button(c, text="set to 1", command=make_1)
        set_1.grid(column=2, row=0, sticky=S, pady=5, padx=2)
        root.bind('<Up>', make_1)

        set_0 = ttk.Button(c, text="set to 0", command=make_0)
        set_0.grid(column=2, row=0, sticky=S, pady=5, padx=2)
        root.bind('<Down>', make_0)

        set_neg1 = ttk.Button(c, text="set to -1", command=make_neg1)
        set_neg1.grid(column=2, row=0, sticky=S, pady=5, padx=2)
        root.bind('1', make_neg1)

        right_btn = ttk.Button(c, text="right", command=move_right)
        right_btn.grid(column=1, row=0, sticky=N, pady=0, padx=0)
        root.bind('<Right>', move_right)

        left_btn = ttk.Button(c, text="left", command=move_left)
        left_btn.grid(column=0, row=0, sticky=S, pady=0, padx=3)
        root.bind('<Left>', move_left)

        switch_lab_btn = ttk.Button(c, text="switch label", command=switch_label)
        switch_lab_btn.grid(column=2, row=0, sticky=S, pady=5, padx=5)
        root.bind('`', switch_label)
        #     root.bind('<1>',switch_label)#click mouse

        quit_btn = ttk.Button(c, text="SAVE", command=save_foo)
        quit_btn.grid(column=3, row=0, sticky=N, pady=5, padx=2)

        quit_btn = ttk.Button(c, text="QUIT", command=root.destroy)
        quit_btn.grid(column=4, row=0, sticky=N, pady=5, padx=2)

        label_1 = ttk.Label(c)
        label_1.grid(column=0, row=10, sticky=N, pady=5, padx=5, columnspan=10, rowspan=1)

        text = Label(c, text=str(im_ind) + ' of ' + str(len(LABELS)))
        text.place(x=100, y=0)
        root.mainloop()
