import random

import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
import six

#stable version1
class ImagePool():
    def __init__(self, pool_size, image_size=256, ch=3, gpu = -1):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            # self.images = []
            self.gpu = gpu
            import numpy
            import cupy
            xp = numpy if gpu < 0 else cupy
            self.images = xp.zeros((self.pool_size, ch, image_size, image_size)).astype("f")

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        xp = chainer.cuda.get_array_module(images)
        for image in images:
            # image = xp.expand_dims(image, axis=0)
            if self.num_imgs < self.pool_size:
                self.images[self.num_imgs] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
                self.num_imgs = self.num_imgs + 1
                # self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = xp.array(chainer.cuda.to_cpu(self.images[random_id]))
                    self.images[random_id] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = xp.stack(return_images)
        return return_images

    def serialize(self, serializer):
        self.gpu = serializer('gpu', self.gpu)
        self.num_imgs = serializer('num_imgs', self.num_imgs)
        self.images = serializer('images', self.images)

#stable2
# class ImagePool():
#     def __init__(self, pool_size, image_size=256, ch=3, gpu = -1):
#         self.pool_size = pool_size
#         if self.pool_size > 0:
#             self.num_imgs = 0
#             # self.images = []
#             self.gpu = gpu
#             import numpy
#             import cupy
#             xp = numpy if gpu < 0 else cupy
#             self.images = xp.zeros((self.pool_size, ch, image_size, image_size)).astype("f")
#
#     def query(self, images):
#         if self.pool_size == 0:
#             return images
#         return_images = []
#         xp = chainer.cuda.get_array_module(images)
#         for image in images:
#             # image = xp.expand_dims(image, axis=0)
#             if self.num_imgs < self.pool_size:
#                 self.images[self.num_imgs] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
#                 self.num_imgs = self.num_imgs + 1
#                 # self.images.append(image)
#                 return_images.append(image)
#             else:
#                 p = random.uniform(0, 1)
#                 if p > 0.5:
#                     random_id = random.randint(0, self.pool_size - 1)
#                     # tmp = xp.array(chainer.cuda.to_cpu(self.images[random_id]))
#                     self.images[random_id] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
#                     return_images.append(image)
#                 else:
#                     random_id = random.randint(0, self.pool_size - 1)
#                     return_images.append(xp.asarray(self.images[random_id]))
#         return_images = xp.stack(return_images)
#         return return_images
#
#     def serialize(self, serializer):
#         self.gpu = serializer('gpu', self.gpu)
#         self.num_imgs = serializer('num_imgs', self.num_imgs)
#         self.images = serializer('images', self.images)

#stable3
# class ImagePool():
#     def __init__(self, pool_size, image_size=256, ch=3, gpu = -1):
#         self.pool_size = pool_size
#         if self.pool_size > 0:
#             self.num_imgs = 0
#             # self.images = []
#             self.gpu = gpu
#             import numpy
#             import cupy
#             xp = numpy if gpu < 0 else cupy
#             self.images = xp.zeros((self.pool_size, ch, image_size, image_size)).astype("f")
#
#     def query(self, images):
#         if self.pool_size == 0:
#             return images
#         return_images = []
#         xp = chainer.cuda.get_array_module(images)
#         fill_flag = False
#         replace_flag = np.zeros(len(images))
#         replace_flag[0:len(images)//2] = 1
#         replace_flag = replace_flag[np.random.permutation(len(images))]
#         replace_count = 0
#         replace_indices = np.random.permutation(self.pool_size)
#         pickbuf_count = 1
#
#         for i, image in enumerate(images):
#             # image = xp.expand_dims(image, axis=0)
#             if self.num_imgs < self.pool_size:
#                 self.images[self.num_imgs] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
#                 self.num_imgs = self.num_imgs + 1
#                 # self.images.append(image)
#                 return_images.append(image)
#                 fill_flag = True
#             else:
#                 if fill_flag:
#                     return_images.append(image)
#                 else:
#                     if replace_flag[i] == 1:
#                         random_id = replace_indices[replace_count]
#                         # tmp = xp.array(chainer.cuda.to_cpu(self.images[random_id]))
#                         self.images[random_id] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
#                         return_images.append(image)
#                         replace_count +=1
#                     else:
#                         random_id = replace_indices[-pickbuf_count]
#                         return_images.append(xp.asarray(self.images[random_id]))
#                         pickbuf_count += 1
#         return_images = xp.stack(return_images)
#         return return_images
#
#     def serialize(self, serializer):
#         self.gpu = serializer('gpu', self.gpu)
#         self.num_imgs = serializer('num_imgs', self.num_imgs)
#         self.images = serializer('images', self.images)

# class ImagePool():
#     def __init__(self, pool_size, image_size=256, ch=3, gpu = -1):
#         self.pool_size = pool_size
#         if self.pool_size > 0:
#             self.num_imgs = 0
#             # self.images = []
#             self.gpu = gpu
#             import numpy
#             import cupy
#             xp = numpy if gpu < 0 else cupy
#             self.images = xp.zeros((self.pool_size, ch, image_size, image_size)).astype("f")
#
#     def query(self, images):
#         if self.pool_size == 0:
#             return images
#         return_images = []
#         xp = chainer.cuda.get_array_module(images)
#         # fill_flag = False
#         # replace_flag = np.zeros(len(images))
#         # replace_flag[0:len(images)//2] = 1
#         # replace_flag = replace_flag[np.random.permutation(len(images))]
#         # replace_count = 0
#         # replace_indices = np.random.permutation(self.pool_size)
#         # pickbuf_count = 1
#         indices_images_random = np.random.permutation(len(images))
#         num_use_buf = min(self.num_imgs, len(images)//2)
#         num_fill = min(self.pool_size-self.num_imgs,len(images)-num_use_buf)
#         num_replace = min(len(images)-num_use_buf-num_fill, self.num_imgs, len(images)-len(images)//2)
#         indices_buf_use = np.random.choice(self.num_imgs,num_use_buf,replace=False)
#         buf_use = self.images[indices_buf_use].copy()
#         indices_buf_replace = np.random.choice(self.num_imgs, num_replace, replace=False)
#         num_replaced = 0
#
#         for i in indices_images_random:
#             # image = xp.expand_dims(image, axis=0)
#             image = images[i]
#             if self.num_imgs < self.pool_size:
#                 self.images[self.num_imgs] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
#                 self.num_imgs = self.num_imgs + 1
#                 # self.images.append(image)
#                 return_images.append(image)
#             else:
#                 if num_replaced < num_replace:
#                     random_id = indices_buf_replace[num_replaced]
#                     self.images[random_id] = chainer.cuda.to_cpu(image) if self.gpu == -1 else image
#                     return_images.append(image)
#                     num_replaced +=1
#
#         return_images = xp.stack(return_images)
#         return return_images
#
#     def serialize(self, serializer):
#         self.gpu = serializer('gpu', self.gpu)
#         self.num_imgs = serializer('num_imgs', self.num_imgs)
#         self.images = serializer('images', self.images)

class HistoricalBuffer():
    def __init__(self, buffer_size=50, image_size=256, image_channels=3, gpu = -1):
        self._buffer_size = buffer_size
        self._img_size = image_size
        self._img_ch = image_channels
        self._cnt = 0
        self.gpu = gpu
        import numpy
        import cupy
        xp = numpy if gpu < 0 else cupy
        self._buffer = xp.zeros((self._buffer_size, self._img_ch, self._img_size, self._img_size)).astype("f")

    def query(self, data, prob=0.5):
        if self._buffer_size == 0:
            return data
        xp = chainer.cuda.get_array_module(data)

        if len(data) == 1:
            if self._cnt < self._buffer_size:
                self._buffer[self._cnt,:] = chainer.cuda.to_cpu(data[0,:]) if self.gpu == -1 else data[0,:]
                self._cnt += 1
                return data
            else:
                if np.random.rand() > prob:
                    self._buffer[np.random.randint(self._cnt), :] = chainer.cuda.to_cpu(data[0,:]) if self.gpu == -1 else data[0,:]
                    return data
                else:
                    return xp.expand_dims(xp.asarray(self._buffer[np.random.randint(self._cnt),:]),axis=0)
        else:
            use_buf = len(data) // 2
            indices_rand = np.random.permutation(len(data))

            avail_buf = min(self._cnt, use_buf)
            if avail_buf > 0:
                indices_use_buf = np.random.choice(self._cnt,avail_buf,replace=False)
                data[indices_rand[-avail_buf:],:] = xp.asarray(self._buffer[indices_use_buf,:])
            room_buf = self._buffer_size - self._cnt
            n_replace_buf = min(self._cnt,len(data)-avail_buf-room_buf)
            if n_replace_buf > 0:
                indices_replace_buf = np.random.choice(self._cnt,n_replace_buf,replace=False)
                self._buffer[indices_replace_buf,:] =  chainer.cuda.to_cpu(data[indices_rand[-avail_buf-n_replace_buf:-avail_buf],:]) \
                    if self.gpu == -1 else data[indices_rand[-avail_buf-n_replace_buf:-avail_buf],:]
            if room_buf > 0:
                n_fill_buf = min(room_buf, len(data)-avail_buf)
                self._buffer[self._cnt:self._cnt+n_fill_buf,:] = chainer.cuda.to_cpu(data[indices_rand[0:n_fill_buf],:]) \
                    if self.gpu == -1 else data[indices_rand[0:n_fill_buf],:]
                self._cnt += n_fill_buf
            return data

    def serialize(self, serializer):
        self._cnt = serializer('cnt', self._cnt)
        self.gpu = serializer('gpu', self.gpu)
        self._buffer = serializer('buffer', self._buffer)


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        super(Updater, self).__init__(*args, **kwargs)
        self._lambda_A = params['lambda_A']
        self._lambda_B = params['lambda_B']
        self._lambda_id = params['lambda_identity']
        self._lrdecay_start = params['lrdecay_start']
        self._lrdecay_period = params['lrdecay_period']
        self._image_size = params['image_size']
        self._max_buffer_size = params['buffer_size']
        self._dataset = params['dataset']
        self._batch_size = params['batch_size']
        self._iter = 0
        self.xp = self.gen_g.xp
        self._buffer_x = ImagePool(self._max_buffer_size, self._image_size)
        self._buffer_y = ImagePool(self._max_buffer_size, self._image_size)
        # self._buffer_x = HistoricalBuffer(self._max_buffer_size, self._image_size)
        # self._buffer_y = HistoricalBuffer(self._max_buffer_size, self._image_size)
        self.init_alpha = self.get_optimizer('gen_g').alpha

    def loss_func_rec_l1(self, x_out, t):
        return F.mean_absolute_error(x_out, t)

    def loss_func_adv_dis_fake(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 0.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def loss_func_adv_dis_real(self, y_real):
        target = Variable(
            self.xp.full(y_real.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_real, target)

    def loss_func_adv_gen(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def update_core(self):
        #debug
        # import tracemalloc
        # snapshot1 = tracemalloc.take_snapshot()
        # top_stats = snapshot1.statistics('lineno')
        # with open("tracemalloc_raw.log", 'a') as f:
        #     print("[ Top 10 ]",file=f)
        #     for stat in top_stats[:10]:
        #         print(stat,file=f)


        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')
        self._iter += 1
        if self.is_new_epoch and self.epoch >= self._lrdecay_start:
            decay_step = self.init_alpha / self._lrdecay_period
            print('lr decay', decay_step)
            if opt_g.alpha > decay_step:
                opt_g.alpha -= decay_step
            if opt_f.alpha > decay_step:
                opt_f.alpha -= decay_step
            if opt_x.alpha > decay_step:
                opt_x.alpha -= decay_step
            if opt_y.alpha > decay_step:
                opt_y.alpha -= decay_step
        batch_x = self.get_iterator('main').next()
        batch_y = self.get_iterator('train_B').next()

        x = Variable(self.converter(batch_x, self.device))
        y = Variable(self.converter(batch_y, self.device))

        x_y = self.gen_g(x)
        x_y_copy = Variable(self._buffer_y.query(x_y.data))
        x_y_x = self.gen_f(x_y)

        y_x = self.gen_f(y)
        y_x_copy = Variable(self._buffer_x.query(y_x.data))
        y_x_y = self.gen_g(y_x)

        loss_gen_g_adv = self.loss_func_adv_gen(self.dis_y(x_y))
        loss_gen_f_adv = self.loss_func_adv_gen(self.dis_x(y_x))

        loss_cycle_x = self._lambda_A * self.loss_func_rec_l1(x_y_x, x)
        loss_cycle_y = self._lambda_B * self.loss_func_rec_l1(y_x_y, y)
        loss_gen = loss_gen_g_adv + loss_gen_f_adv + loss_cycle_x + loss_cycle_y

        if self._lambda_id > 0:
            loss_id_x = self._lambda_id * F.mean_absolute_error(x,
                                                                self.gen_f(x))
            loss_id_y = self._lambda_id * F.mean_absolute_error(y,
                                                                self.gen_g(y))
            loss_gen = loss_gen + loss_id_x + loss_id_y

        self.gen_f.cleargrads()
        self.gen_g.cleargrads()
        loss_gen.backward()
        opt_f.update()
        opt_g.update()

        loss_dis_y_fake = self.loss_func_adv_dis_fake(self.dis_y(x_y_copy))
        loss_dis_y_real = self.loss_func_adv_dis_real(self.dis_y(y))
        loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5
        self.dis_y.cleargrads()
        loss_dis_y.backward()
        opt_y.update()

        loss_dis_x_fake = self.loss_func_adv_dis_fake(self.dis_x(y_x_copy))
        loss_dis_x_real = self.loss_func_adv_dis_real(self.dis_x(x))
        loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5
        self.dis_x.cleargrads()
        loss_dis_x.backward()
        opt_x.update()

        chainer.report({'loss': loss_dis_x}, self.dis_x)
        chainer.report({'loss': loss_dis_y}, self.dis_y)
        chainer.report({'loss_cycle': loss_cycle_y}, self.gen_g)
        chainer.report({'loss_cycle': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_gen': loss_gen_g_adv}, self.gen_g)
        chainer.report({'loss_gen': loss_gen_f_adv}, self.gen_f)

        if self._lambda_id > 0:
            chainer.report({'loss_id': loss_id_y}, self.gen_g)
            chainer.report({'loss_id': loss_id_x}, self.gen_f)

        #debug
        # snapshot2 = tracemalloc.take_snapshot()
        # top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        # with open("tracemalloc_diff.log",'a') as f:
        #     print("[ Top 10 differences ]",file=f)
        #     for stat in top_stats[:10]:
        #         print(stat,file=f)

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)

        self._buffer_x.serialize(serializer['buffer_x'])
        self._buffer_y.serialize(serializer['buffer_y'])
