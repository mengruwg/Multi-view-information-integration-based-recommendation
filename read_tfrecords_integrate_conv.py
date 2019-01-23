# -*- coding:utf-8 -*-

## This file is for prepare tfrecords data and read in the data in training and testing period

import tensorflow as tf
#import cv2
import types
product_size = 3200
des_feature_size = 1461
review_feature_size = 1894
batch_size = 10
customer_no = 27341

def encode_to_tfrecords(lable_file,data_root,new_name='preprocess.tfrecords',resize=None):
    writer=tf.python_io.TFRecordWriter(new_name)
    num_example=0
    with open(lable_file,'r') as f:
        for l in f.readlines():
            data = l.split(",")
            #print len(data)
            label = [int(i) for i in data[0:1]]
            img = cv2.imread(data[1]) 
            des = [float(i) for i in data[2:des_feature_size+2]]
            rev = [float(i) for i in data[des_feature_size+2:des_feature_size+review_feature_size+2]]

            example=tf.train.Example(features=tf.train.Features(feature={
                'img':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                'des':tf.train.Feature(float_list=tf.train.FloatList(value=des)),
                'rev':tf.train.Feature(float_list=tf.train.FloatList(value=rev)),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=label))
            }))
            serialized=example.SerializeToString()
            writer.write(serialized)
            num_example+=1
    print (lable_file, "total sample:",num_example)
    writer.close()

def encode_to_tfrecords2(lable_file,data_root,new_name='preprocess.tfrecords',resize=None):
    #writer=tf.python_io.TFRecordWriter(data_root+'/'+new_name)
    writer=tf.python_io.TFRecordWriter(new_name)
    num_example=0
    with open(lable_file,'r') as f:
        for l in f.readlines():
            data = l.split(",")
            des = [float(i) for i in data[0:3]]
            #print len(data)

            #label=int(l[1])

            example=tf.train.Example(features=tf.train.Features(feature={
                #'label':tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                #'img':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                'des':tf.train.Feature(float_list=tf.train.FloatList(value=des)),
                #'rev':tf.train.Feature(float_list=tf.train.FloatList(value=rev)),
                #'label':tf.train.Feature(int64_list=tf.train.Int64List(value=label))
            }))
            serialized=example.SerializeToString()
            writer.write(serialized)
            num_example+=1
    print (lable_file, "Total sample:",num_example)
    writer.close()
#encode_to_tfrecords('28.integrate_feature.csv','',new_name='integrate_feature.tfrecords',resize=None)
#encode_to_tfrecords2('random_cog_value.csv','',new_name='random_cog_value.tfrecords',resize=None)

def decode_from_tfrecords(filename,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)
    #_,serialized=reader.read(filename)
    example=tf.parse_single_example(serialized,features={
        #'height':tf.FixedLenFeature([],tf.int64),
        #'width':tf.FixedLenFeature([],tf.int64),
        #'label':tf.FixedLenFeature([],tf.string),
        'user':tf.FixedLenFeature([1],tf.int64),
        'img':tf.FixedLenFeature([],tf.string),
        'des':tf.FixedLenFeature([des_feature_size],tf.float32),
        'rev':tf.FixedLenFeature([review_feature_size],tf.float32),
        'label':tf.FixedLenFeature([1],tf.int64)
    })

    img=tf.decode_raw(example['img'],tf.uint8)
    img = tf.reshape(img,[50,38,3])
    #img = tf.reduce_mean(img,2)
    #img = tf.reshape(img,[1900,])
    img = tf.cast(img, tf.float32)/255.0
    des = example['des']
    rev = example['rev']
    label = tf.cast(example['label'], tf.int32)
    user = tf.cast(example['user'],tf.int32)

    return user,img,des,rev,label

def decode_from_tfrecords2(filename,num_epoch=None):
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数
    reader=tf.TFRecordReader()
    _,serialized=reader.read(filename_queue)
    #_,serialized=reader.read(filename)
    example=tf.parse_single_example(serialized,features={
        #'height':tf.FixedLenFeature([],tf.int64),
        #'width':tf.FixedLenFeature([],tf.int64),
        #'label':tf.FixedLenFeature([],tf.string),
        'img':tf.FixedLenFeature([],tf.string),
        'des':tf.FixedLenFeature([des_feature_size],tf.float32),
        'rev':tf.FixedLenFeature([review_feature_size],tf.float32),
        'label':tf.FixedLenFeature([1],tf.int64)
    })

    img=tf.decode_raw(example['img'],tf.uint8)
    img = tf.reshape(img,[50,38,3])
    #img = tf.reduce_mean(img,2)
    #img = tf.reshape(img,[1900,])
    img = tf.cast(img, tf.float32)/255.0
    des = example['des']
    rev = example['rev']
    label = tf.cast(example['label'], tf.int32)
    '''print img
    print des
    print rev
    print label'''
    return img,des,rev,label

#img,des,rev,label = decode_from_tfrecords('integrate_feature.tfrecords')

def get_batch(batch_size):
    #distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])
    #distorted_image = decode_from_tfrecords('description.tfrecords')
    user,img,des,rev,label = decode_from_tfrecords('random_integrate_4.tfrecords')

    user,img,des,rev,label = tf.train.shuffle_batch([user,img,des,rev,label],batch_size=batch_size,num_threads=16,capacity=10000,min_after_dequeue=2000)
    #img,des,rev,label = tf.train.batch([img,des,rev,label],batch_size=batch_size)

    #tf.image_summary('images', images)
    #print images.get_shape()
    print (user,img,des,rev,label)
    return user,img,des,rev,label

#get_batch(batch_size)

def get_random_batch(batch_size):
        #数据扩充变换
    #distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪
    #distorted_image = decode_from_tfrecords('description.tfrecords')
    img,des,rev,label = decode_from_tfrecords2('integrate_feature.tfrecords')
    #distorted_image = tf.image.random_flip_up_down(image)#上下随机翻转
    #distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化
    #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化

    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
    img,des,rev,label = tf.train.shuffle_batch([img,des,rev,label],batch_size=batch_size,
                                                 num_threads=16,capacity=3000,min_after_dequeue=1000)
    #img,des,rev,label = tf.train.batch([img,des,rev,label],batch_size=batch_size)
    # 调试显示
    #tf.image_summary('images', images)
    #print images.get_shape()
    #print img,des,rev,label
    return img,des,rev,label

def get_evaluate_batch(batch_size):
        #数据扩充变换
    #distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪
    #distorted_image = decode_from_tfrecords('description.tfrecords')
    img, des,rev,label = decode_from_tfrecords2('integrate_feature.tfrecords')
    #distorted_image = tf.image.random_flip_up_down(image)#上下随机翻转
    #distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化
    #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化

    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
    #images,label = tf.train.shuffle_batch([distorted_image,label],batch_size=batch_size,
    #                                             num_threads=16,capacity=3000,min_after_dequeue=1000)
    img,des,rev,label = tf.train.batch([img,des,rev,label],batch_size=batch_size)
    # 调试显示
    #tf.image_summary('images', images)
    #print images.get_shape()
    #print img,des,rev,label
    return img,des,rev


def test():
    #encode_to_tfrecords("data/train.txt","data",(100,100))
    #image = decode_from_tfrecords('/Users/kanetsu/full/data.tfrecords')
    #des,rev,m = get_batch(batch_size)#batch 生成测试
    user,img,des,rev,label = get_batch(batch_size)#batch 生成测试
    #init=tf.initialize_all_variables()
    #test = tf.concat([des,rev],1)
    init=tf.initialize_all_variables()
    total_batch = int(product_size/batch_size)
    with tf.Session() as session:
        session.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        for l in range(total_batch):
            #image_np,label_np=session.run([image,label])#每调用run一次，那么
            user= session.run([user])
            print ('batch no:', l)
      coord.request_stop()#queue需要关闭，否则报错
        coord.join(threads)


#test()



