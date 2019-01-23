# -*- coding:utf-8 -*-
import tensorflow as tf  
import numpy as np  
#import matplotlib.pyplot as plt  
#import read_tfrecords_rating
import read_tfrecords_integrate_conv
import os.path
import types
import math
import csv
import sys
from datetime import datetime 

  
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'hifa_200',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'hifa_200',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# some hyperparameters and structure parameters
isTrain = True
N = 50
learning_rate = 0.001  
training_epochs = 82
batch_size = 1
evaluate_batch_size = 3200
display_step = 1  
evaluate_step = 81

learning_rate = 0.001
lambda_bias = 0.001
lambda_embedding = 0.001
lambda_latent = 0.1

height = 50
width = 38
channel = 3

fm1 = 64 
fm2 = 64
fc = 100 

img_dim = 1900
img_l1 = 256
img_l2 = 128

des_dim = 1461
des_l1 = 400
des_l2 = fc

rev_dim = 1894
rev_l1 = 400
rev_l2 = fc

img_ae = 1
des_ae = 1
rev_ae = 1

rating_no = 16281
customer_no = 27341
product_no = 3200
factor_no = 180
latent_factor_no = 20

# for image representation
def image_inference(input,weights,decode_weights,image_bias): 
  layer_1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(input, weights['weights_1'], [1, 1, 1, 1], padding='SAME'),image_bias['bias_1']))
  layer_2 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(layer_1, weights['weights_2'], [1, 1, 1, 1], padding='SAME'),image_bias['bias_2']))
  layer_2 = tf.reshape(layer_2,[-1,height*width*fm2])
  layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['weights_3']),image_bias['bias_3']))  
  layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, decode_weights['weights_4']),image_bias['bias_4']))
  layer_4 = tf.reshape(layer_4,[-1,height,width,fm2])
  layer_5 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(layer_4, decode_weights['weights_5'], [1, 1, 1, 1], padding='SAME'),image_bias['bias_5']))
  layer_6 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(layer_5, decode_weights['weights_6'],[1, 1, 1, 1], padding='SAME'),image_bias['bias_6']))
  return layer_6

def extract_image_feature(input,weights,decode_weights,image_bias):
  layer_1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(input, weights['weights_1'], [1, 1, 1, 1], padding='SAME'),image_bias['bias_1']))
  layer_2 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(layer_1, weights['weights_2'], [1, 1, 1, 1], padding='SAME'),image_bias['bias_2']))
  layer_2 = tf.reshape(layer_2,[-1,height*width*fm2])
  layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['weights_3']),image_bias['bias_3']))  
  return layer_3

def img_encoder(x,weights,biaes):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                               biaes['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                               biaes['encoder_b2']))
    return layer_2

def img_decoder(x,weights,biaes):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'],perm = [1,0])),
                               biaes['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h1'],perm = [1,0])),
                               biaes['decoder_b2']))
    return layer_2

# for description representation
def des_encoder(x,weights,biaes):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                               biaes['encoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                               biaes['encoder_b2']))  
    return layer_2  

 
def des_decoder(x,weights,biaes):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'],perm = [1,0])),  
                               biaes['decoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h1'],perm = [1,0])),  
                               biaes['decoder_b2']))  
    return layer_2  


# for review representation
def rev_encoder(x,weights,biaes):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),  
                               biaes['encoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),  
                               biaes['encoder_b2']))  
    return layer_2  


 
def rev_decoder(x,weights,biaes):  
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['encoder_h2'],perm = [1,0])),  
                               biaes['decoder_b1']))  
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['encoder_h1'],perm = [1,0])),  
                               biaes['decoder_b2']))  
    return layer_2 


def train(optimizer,cost,merged_sumary_op,saver):
  init = tf.global_variables_initializer()  
  sess = tf.Session()
  sess.run(init)  

  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess,ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    global_step = int(global_step)
  else:
    global_step = 0

  coord = tf.train.Coordinator()  
  threads = tf.train.start_queue_runners(coord=coord,sess=sess) 

  total_batch = int(rating_no/batch_size)

  #summary_writer = tf.summary.FileWriter('log', sess.graph)
  previous_cost = 0
  for epoch in range(global_step,training_epochs):  
    total_cost = 0
    for i in range(total_batch):  
        #_, c, summary_str = sess.run([optimizer, cost, merged_summary_op])
        _, c = sess.run([optimizer, cost])
        #summary_writer.add_summary(summary_str, total_batch*epoch+i+1)
        print("Time: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')),'batch %d' %(i+1), 'cost: %.9f' % c) 
        total_cost = total_cost + c

    if epoch % display_step == 0:  
        average_cost = total_cost / total_batch
        percentage = (previous_cost - average_cost) / average_cost
        if percentage < 0.01:
          break
        previous_cost = average_cost
        print("Time: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')),"Epoch %04d" % (epoch+1), "average cost: %.9f" % (average_cost))  

  checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
  saver.save(sess, checkpoint_path, global_step=training_epochs)
  print("Optimization Finished!") 
  coord.request_stop()  
  coord.join(threads)

# calculate evaluation metric: AUC and hit ratio

def evaluate_hit(f,user_m,item_m,item_bias, A,image_feat,des_feat,review_feat,P,saver):
  init = tf.global_variables_initializer()  
  sess = tf.Session()
  sess.run(init)  
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess,ckpt.model_checkpoint_path)
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    global_step = int(global_step)
  else:
    global_step = 0

  coord = tf.train.Coordinator()  
  threads = tf.train.start_queue_runners(coord=coord,sess=sess) 
  
  user_m, item_m, item_bias, A,P,image_feat,des_feat,review_feat = sess.run([user_m, item_m,item_bias,A,P,image_feat,des_feat,review_feat]) 


  '''np.savetxt('user_m.txt',user_m)
  np.savetxt('item_factor.txt',item_factor)'''
  coord.request_stop()  
  coord.join(threads)
  print ("Evaluation begins:")

  train = {}
  null = []
  for i in range(customer_no):
    train[i] = null
  train_table = csv.reader(open('3.train.csv','r')) 
  for line in train_table:
    customer = int(line[0])
    item = int(line[1])
    train[customer] = train[customer] + [item]
   
  test = csv.reader(open('4.test.csv','r'))

  #test = csv.reader(open('6_test.csv','r'))
  #test = csv.reader(open('17.less_than_5_testset.csv','r'))
  total_hit_30 = 0
  total_hit_50 = 0
  total_hit_100 = 0
  total_hit_150 = 0
  total_hit_200 = 0
  sum_count = 0 
  for count, row in enumerate(test):
    if count >= 0:
      print (count)
      U = {}
      user = int(row[0])
      pos_item = int(row[1])
      m = 0  
      for i in range(product_no):
        if i not in train[user]:
          m += 1
          pos_attention = A[user]
         
          img = image_feat[i] * pos_attention[0]
          des = des_feat[i] * pos_attention[1]
          rev = review_feat[i] * pos_attention[2]
       
          item_factor = np.concatenate((np.dot(np.concatenate((img,des,rev)),P),item_m[i]))
          #content_factor = np.dot(np.concatenate((img,des,rev)),P)
          #item_factor = np.concatenate((content_factor,item_m[pos_item]))
          score = np.dot(user_m[user],item_factor)+item_bias[i]
          U[i] = score
        #else:
          #U[i] = -9999
      rec = sorted(U.items(),key = lambda item:item[1], reverse = True)
      rec_list = []
      length = len(rec)
      #print "length", length
      for i in range(length):
        rec_list.append(rec[i][0])

      position = rec_list.index(pos_item)
      sum_count += float(m - position - 1) / float(m - 1)
      if position < 30:
        print ("hit!")
        total_hit_30 += 1
      if position < 50:
        total_hit_50 += 1
      if position < 100:
        total_hit_100 += 1
      if position < 150:
        total_hit_150 += 1
      if position < 200:
        total_hit_200 += 1
      #sum_count += float(m - position - 1) / float(m - 1)
  hit_ratio_30 = total_hit_30/float(count+1)
  hit_ratio_50 = total_hit_50/float(count+1)
  hit_ratio_100 = total_hit_100/float(count+1)
  hit_ratio_150 = total_hit_150/float(count+1)
  hit_ratio_200 = total_hit_200/float(count+1)
  auc = sum_count/float(count+1)
  #hit_ratio = total_hit/float(count+1)
 
  print ('auc: %.9f' % (auc))
  #print 'hit ratio: %.9f' % (hit_ratio)
  #return auc,hit_ratio
  #f.write(str(hit_ratio)+'\n')
  print ('hit_30 : %.9f' % (hit_ratio_30))
  print ('hit_50 : %.9f' % (hit_ratio_50))
  print ('hit_100 : %.9f' % (hit_ratio_100))
  print ('hit_150 : %.9f' % (hit_ratio_150))
  print ('hit_200 : %.9f' % (hit_ratio_200))
  print ('auc: %.9f' % (auc))
  return auc, hit_ratio_30, hit_ratio_50,hit_ratio_100,hit_ratio_150,hit_ratio_200

def test():  
  f = csv.writer(open('hifa_new.csv','a'))
  f2 = csv.writer(open('cost.csv','a'))
  f.writerow(['learning_rate','bias','embedding','latent','factor_no','fm1','fm2','fc','des_l1','des_l2','rev_l1','rev_l2','img_ae','des_ae','rev_ae','auc','hit','epoch'])

  tf.set_random_seed(1234)
  #lambda_bias = 0.01
  #lambda_embedding = 0.01
  #lambda_latent = 1

# initialize network weight parameters

  weights = {  
  'weights_1': tf.Variable(tf.random_normal([3,3,3,fm1],1e-4)),  
  'weights_2': tf.Variable(tf.random_normal([3,3,fm1,fm2],1e-4)),
  'weights_3': tf.Variable(tf.random_normal([(height)*(width)*fm2,fc],1)),
  } 
  decode_weights = {
  'weights_4': tf.transpose(weights['weights_3'],perm=[1,0]),
  'weights_5': tf.transpose(weights['weights_2'],perm=[0,1,3,2]),
  'weights_6': tf.transpose(weights['weights_1'],perm=[0,1,3,2]),
  } 
  image_bias = {  
  'bias_1': tf.Variable(tf.zeros([fm1])), 
  'bias_2': tf.Variable(tf.zeros([fm2])),
  'bias_3': tf.Variable(tf.zeros([fc])),
  'bias_4': tf.Variable(tf.zeros([(height)*(width)*fm2])),
  'bias_5': tf.Variable(tf.zeros([fm1])),
  'bias_6': tf.Variable(tf.zeros([3])), 
  } 
  img_weights = {
  'encoder_h1': tf.Variable(tf.random_normal([img_dim, img_l1])),
  'encoder_h2': tf.Variable(tf.random_normal([img_l1, img_l2])),
  #'decoder_h1': tf.Variable(tf.random_normal([des_l2, des_l1])),  
  #'decoder_h2': tf.Variable(tf.random_normal([des_l1, des_dim])),  
  }
  img_bias = {
  'encoder_b1': tf.Variable(tf.random_normal([img_l1])),
  'encoder_b2': tf.Variable(tf.random_normal([img_l2])),
  'decoder_b1': tf.Variable(tf.random_normal([img_l1])),
  'decoder_b2': tf.Variable(tf.random_normal([img_dim])),
  }
  des_weights = {  
  'encoder_h1': tf.Variable(tf.random_normal([des_dim, des_l1])),  
  'encoder_h2': tf.Variable(tf.random_normal([des_l1, des_l2])),  
  #'decoder_h1': tf.Variable(tf.random_normal([des_l2, des_l1])),  
  #'decoder_h2': tf.Variable(tf.random_normal([des_l1, des_dim])),  
  }  
  des_bias = {  
  'encoder_b1': tf.Variable(tf.random_normal([des_l1])),  
  'encoder_b2': tf.Variable(tf.random_normal([des_l2])),  
  'decoder_b1': tf.Variable(tf.random_normal([des_l1])),  
  'decoder_b2': tf.Variable(tf.random_normal([des_dim])),  
  } 

  rev_weights = {  
  'encoder_h1': tf.Variable(tf.random_normal([rev_dim, rev_l1])),  
  'encoder_h2': tf.Variable(tf.random_normal([rev_l1, rev_l2])),  
  #'decoder_h1': tf.Variable(tf.random_normal([des_l2, des_l1])),  
  #'decoder_h2': tf.Variable(tf.random_normal([des_l1, des_dim])),  
  }  
  rev_bias = {  
  'encoder_b1': tf.Variable(tf.random_normal([rev_l1])),  
  'encoder_b2': tf.Variable(tf.random_normal([rev_l2])),  
  'decoder_b1': tf.Variable(tf.random_normal([rev_l1])),  
  'decoder_b2': tf.Variable(tf.random_normal([rev_dim])),  
  } 

  user_m = tf.Variable(tf.random_normal([customer_no,factor_no+latent_factor_no]))

  #P = tf.Variable(tf.random_normal([fc+des_l2+rev_l2, factor_no]))
  P = tf.Variable(tf.zeros([fc+des_l2+rev_l2, factor_no]))
  #A = tf.Variable(tf.zeros([img_dim + des_dim + rev_dim + factor_no + latent_factor_no, 3]))
  #attention_bias = tf.Variable(tf.zeros([3]))
  A = tf.Variable(tf.random_normal([customer_no, 3]))

  item_m = tf.Variable(tf.random_normal([product_no,latent_factor_no]))
  item_bias = tf.Variable(tf.zeros([product_no]))
  saver = tf.train.Saver(tf.global_variables())


  if isTrain:
    # formulate cost function
    user,M,D,R,label = read_tfrecords_integrate_conv.get_batch(batch_size)
    M_2,D_2,R_2,label_2 = read_tfrecords_integrate_conv.get_random_batch(batch_size)
    
    # = img_encoder(M,img_weights,img_bias)
    #encoder_D_2 = des_encoder(noised_D_2,des_weights,des_bias)
    #decoder_M = img_decoder(encoder_M,img_weights,img_bias) 
    decoder_M = image_inference(M,weights,decode_weights,image_bias)  
    #cost1 = tf.reduce_mean(tf.pow(decoder_M - M, 2)) + (tf.reduce_mean(tf.pow(img_weights['encoder_h2'],2))+ tf.reduce_mean(tf.pow(img_weights['encoder_h1'],2)))
    cost1 = tf.reduce_mean(tf.pow(M - decoder_M, 2)) + tf.reduce_mean(tf.pow(weights['weights_1'],2)) + tf.reduce_mean(tf.pow(weights['weights_2'],2)) + tf.reduce_mean(tf.pow(weights['weights_3'],2))

    noised_D = D + 0*tf.random_normal([batch_size,des_dim])
    noised_D_2 = D_2 + 0*tf.random_normal([batch_size,des_dim])
    encoder_D = des_encoder(noised_D,des_weights,des_bias)
    encoder_D_2 = des_encoder(noised_D_2,des_weights,des_bias)
    decoder_D = des_decoder(encoder_D,des_weights,des_bias)
    #cost2 = tf.reduce_sum(tf.pow(decoder_D - D, 2)) + 0.01*(tf.reduce_sum(tf.pow(des_weights['encoder_h2'],2))+ tf.reduce_sum(tf.pow(des_weights['encoder_h1'],2)))
    cost2 = tf.reduce_mean(tf.pow(decoder_D - D, 2)) + (tf.reduce_mean(tf.pow(des_weights['encoder_h2'],2))+ tf.reduce_mean(tf.pow(des_weights['encoder_h1'],2)))

    noised_R = R + 0*tf.random_normal([batch_size,rev_dim])
    noised_R_2 = R_2 + 0*tf.random_normal([batch_size,rev_dim])
    encoder_R = des_encoder(noised_R,rev_weights,rev_bias)
    encoder_R_2 = des_encoder(noised_R_2,rev_weights,rev_bias)
    decoder_R = des_decoder(encoder_R,rev_weights,rev_bias)
    #cost3 = tf.reduce_sum(tf.pow(decoder_R - R, 2)) + 0.01*(tf.reduce_sum(tf.pow(rev_weights['encoder_h2'],2))+ tf.reduce_sum(tf.pow(rev_weights['encoder_h1'],2)))
    cost3 = tf.reduce_mean(tf.pow(decoder_R - R, 2)) + (tf.reduce_mean(tf.pow(rev_weights['encoder_h2'],2))+ tf.reduce_mean(tf.pow(rev_weights['encoder_h1'],2)))

    #user_list = read_tfrecords_rating.get_batch(batch_size)
    
    #user_code = user_list[0][0]
    user_code = user[0][0]
    user_factor = tf.expand_dims(user_m[user_code,:],-1)
    trans_user_factor = tf.transpose(user_factor,perm = [1,0])

    pos_attention = A[user_code,:]
    neg_attention = A[user_code,:]

    image_feat = extract_image_feature(M,weights,decode_weights,image_bias) * pos_attention[0]
    neg_image_feat = extract_image_feature(M_2,weights,decode_weights,image_bias) * neg_attention[0]
    
    des_feat = encoder_D * pos_attention[1]
    neg_des_feat = encoder_D_2 * neg_attention[1]

    review_feat = encoder_R * pos_attention[2]
    neg_review_feat = encoder_R_2 * neg_attention[2]

    feat = tf.concat([image_feat,des_feat,review_feat],1)
    neg_feat = tf.concat([neg_image_feat,neg_des_feat,neg_review_feat],1)

    item_content_factor = tf.matmul(feat,P)
    neg_item_content_factor = tf.matmul(neg_feat,P)

    item_latent_factor = tf.expand_dims(item_m[label[0][0],:],0)
    neg_item_latent_factor = tf.expand_dims(item_m[label_2[0][0],:],0)
    
    item_factor = tf.concat([item_content_factor,item_latent_factor],1)
    neg_item_factor = tf.concat([neg_item_content_factor,neg_item_latent_factor],1)
    
    pos_item_bias = item_bias[label[0][0]]
    neg_item_bias = item_bias[label_2[0][0]]
    

    cost4 = -tf.log(tf.nn.sigmoid(tf.matmul(item_factor-neg_item_factor, user_factor)+(pos_item_bias-neg_item_bias)))
    cost5 = cost4 + (lambda_bias * tf.reduce_sum(tf.pow(neg_item_bias,2)) + 0.1 * lambda_latent * tf.reduce_sum(tf.pow(neg_item_latent_factor,2))) + lambda_bias * tf.reduce_sum(tf.pow(pos_item_bias,2)) + lambda_embedding * tf.reduce_sum(tf.pow(P,2)) + lambda_latent * tf.reduce_sum(tf.pow(user_factor,2)) + lambda_latent * tf.reduce_sum(tf.pow(item_latent_factor,2))
    
    # 定义代价函数和优化器  
    cost = cost1 +  cost2 + cost3 +  cost5
    #cost = cost4
    #user_latent_factor = user_factor[-latent_factor_no:,:]

    tf.summary.scalar("cost_function", cost)
    merged_summary_op = tf.summary.merge_all()

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)  
    #train(optimizer,cost,merged_summary_op,saver)

    init = tf.global_variables_initializer()  
    sess = tf.Session()
    # with tf.Session() as sess:
    sess.run(init)  
    # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练  

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess,ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      global_step = int(global_step)
    else:
      global_step = 0
    #global_step = 0

    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(coord=coord,sess=sess) 

    total_batch = int(rating_no/batch_size)
    #total_batch = 10
    #summary_writer = tf.summary.FileWriter('log', sess.graph)
    previous_cost = 0 
    
    # optimization loop 
    for epoch in range(global_step,training_epochs):  
      total_cost = 0
      for i in range(total_batch):  
          #_, c, summary_str = sess.run([optimizer, cost, merged_summary_op])
          _, c = sess.run([optimizer, cost])
          #summary_writer.add_summary(summary_str, total_batch*epoch+i+1)
          if (i+1) % 5000 == 0:
            print("Time: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')),'batch %d' %(i+1), 'cost: %.9f' % c) 
          total_cost = total_cost + c

      if epoch % display_step == 0:  
          average_cost = total_cost / total_batch
          percentage =abs(previous_cost - average_cost) / average_cost
          print("Time: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')),"Epoch %04d" % (epoch+1), "average cost: %.9f" % (average_cost))
          f2.writerow([epoch+1]+[average_cost])
          if percentage < 0.0001:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=epoch+1)
            break
          previous_cost = average_cost

          #print("Time: %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')),"Epoch %04d" % (epoch+1), "average cost: %.9f" % (average_cost))  

      if (epoch+1) % evaluate_step == 0:
          #f.write('epoch:'+str(epoch+1)+'\n')
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=epoch+1)
          M,D,R = read_tfrecords_integrate_conv.get_evaluate_batch(evaluate_batch_size)
          #input_vector = tf.concat([M,D,R],1)

          #image_feat = img_encoder(M,img_weights,img_bias)
          image_feat = extract_image_feature(M,weights,decode_weights,image_bias)
          des_feat = des_encoder(D,des_weights,des_bias)
          review_feat = des_encoder(R,rev_weights,rev_bias)
          auc,hit_30,hit_50,hit_100,hit_150,hit_200 = evaluate_hit(f,user_m,item_m,item_bias,A,image_feat,des_feat,review_feat,P,saver)
          f.writerow([learning_rate]+[lambda_bias]+[lambda_embedding]+[lambda_latent]+[factor_no+latent_factor_no]+[fm1]+[fm2]+[fc]+[des_l1]+[des_l2]+[rev_l1]+[rev_l2]+[img_ae]+[des_ae]+[rev_ae]+[auc]+[hit_30]+[epoch+1])
          f.writerow([hit_30]+[hit_50]+[hit_100]+[hit_150]+[hit_200])

          #f.writerow([learning_rate]+[lambda_bias]+[lambda_embedding]+[lambda_latent]+[factor_no+latent_factor_no]+[fm1]+[fm2]+[fc]+[des_l1]+[des_l2]+[rev_l1]+[rev_l2]+[img_ae]+[des_ae]+[rev_ae]+[auc]+[hit]+[epoch+1])

    M,D,R = read_tfrecords_integrate_conv.get_evaluate_batch(evaluate_batch_size)

    image_feat = extract_image_feature(M,weights,decode_weights,image_bias)
    des_feat = des_encoder(D,des_weights,des_bias)
    review_feat = des_encoder(R,rev_weights,rev_bias)
    auc,hit_30,hit_50,hit_100,hit_150,hit_200 = evaluate_hit(f,user_m,item_m,item_bias,A,image_feat,des_feat,review_feat,P,saver)
    f.writerow([learning_rate]+[lambda_bias]+[lambda_embedding]+[lambda_latent]+[factor_no+latent_factor_no]+[fm1]+[fm2]+[fc]+[des_l1]+[des_l2]+[rev_l1]+[rev_l2]+[img_ae]+[des_ae]+[rev_ae]+[auc]+[hit_30]+[global_step])
    f.writerow([hit_30]+[hit_50]+[hit_100]+[hit_150]+[hit_200])


  
    print("Optimization Finished!") 
    coord.request_stop()  
    coord.join(threads) 

#test(0.1,0.1,0.1)
#test(10,10,10)
test()
