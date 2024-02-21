#print("strating blending code")
import tensorflow as tf
from absl import flags
import numpy as np
import cv2
import numpy as np
import skimage.io
#import tensorflow as tf
import os
import random
from absl import logging
from PIL import Image
from misc import tf_blend_uv
import yaml
import tensorflow as tf
from PIL import Image

def image_to_tensorr(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [512, 512])
    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    return image

def image_to_tensor(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [512, 512])
    image = tf.expand_dims(image, axis=0)
    #image = tf.expand_dims(image, axis=-1)    
    
    return image

def main():
    cwd = '/app' #os.getcwd()
    CONFIG_FILE = "/app/config.yml"
    # Read and update configuration from YAML
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    user_choice = config['general']['user_gender'] #input("Is this image male or female? ")
    print(">>>>>>>>>>>>user_choice", config['general']['user_gender'])
    if user_choice=="male":
      print("IN MALE")

      for i in range(3):
        if i==0:

          image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask', 'male','right.png')
          image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','right_00.png')
          base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','front_00.png')
          output_dir_int = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result')
          output_dir = output_dir_int
          #output_dir = os.path.join(cwd, 'data', 'temporary_result', 'left_blended_output')
        


        if i==1:
          image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask','male', 'left.png')
          image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','left_00.png')
          base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result', 'blended_output_0.png')
          #output_dir = os.path.join(cwd, 'data', 'temporary_result', 'right_blended_output')
          output_dir = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result')


        if i==2:
          image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask','male', 'front.png')
        #image_path = os.path.join(cwd, 'data', 'mask', 'front.png')
          image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','front_00.png')
          base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result','blended_output_1.png')
          output_dir = os.path.join(cwd, 'data', 'input')
          #base_uv_path = os.path.join(cwd, 'data', 'temporary_result', 'right_blended_output', 'blended_output.png')

        #graph definition
            # Create TensorFlow graph
        graph = tf.Graph()
        with graph.as_default():
            image_tensorr = image_to_tensorr(image_path)
        
        ###defination of mask
            sess = tf.compat.v1.Session()
            with sess.as_default():
                image_arrayy = image_tensorr.eval()
        # image_tensorr = image_to_tensorr(image_path)
        # image_arrayy = image_tensorr.eval()
            image_listt = image_arrayy.tolist()
            ptt = np.squeeze(np.array(image_listt))
            uv_test_mask = np.squeeze(np.array(image_listt))
            uv_test_maskk = tf.expand_dims(uv_test_mask, axis=0) 
            uv_test_maskk = tf.expand_dims(uv_test_maskk, axis=-1)

            uv_test_maskk = tf.cast(uv_test_maskk, dtype=tf.float32)
      

     #defination of front image
        
            image_tensor = image_to_tensor(image_pathh)
            sess = tf.compat.v1.Session()
            with sess.as_default():
                image_array = image_tensor.eval()
            image_list = image_array.tolist()
            pt=np.squeeze(np.array(image_list)) 
            front_uv_batch_test = np.squeeze(np.array(image_list))
        #uv_batch_test = tf.expand_dims(front_uv_batch_test, axis=0)
            uv_batch_test = tf.expand_dims(tf.cast(front_uv_batch_test, dtype=tf.float32), axis=0)


        ###################

     ########defination of base image
            base_uv = Image.open(base_uv_path).resize((512, 512))
            base_uv = np.asarray(base_uv, np.float32) / 255
            base_uv_batch = tf.constant(base_uv[np.newaxis, ...], name="base_uv")
            base_uv_batch = tf.cast(base_uv_batch, dtype=tf.float32)
        ##########
        



            uv_batch = tf_blend_uv(
              base_uv_batch, uv_batch_test, uv_test_maskk, match_color=True
          )
            uv_batch = tf.identity(uv_batch, name="uv_tex")
            uv_tex_res = uv_batch[0]

        with tf.compat.v1.Session(graph=graph) as sess:
        # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())

        


        # Run the session and get the output
            uv_tex_res_val = sess.run(uv_tex_res)

    # Convert the NumPy array to PIL Image and save it
        uv_tex_res_image = Image.fromarray((uv_tex_res_val * 255).astype(np.uint8))
                #filename = "portrait.png"
        if i == 2:
           filename = "portrait.png"
        else:   
           filename = "blended_output_{}.png".format(i)
      
        uv_tex_res_image.save(os.path.join(output_dir, filename)) 
         
    else:
      #print("IN FEMALE")
      for i in range(3):
        #print("jhjhj",i)
        if i==0:

          image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask', 'female','right.png')
          image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','right_00.png')
          base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','front_00.png')
          output_dir = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result')
          #output_dir = os.path.join(cwd, 'data', 'temporary_result', 'left_blended_output')
        


        if i==1:
          image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask','female', 'left.png')
          image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','left_00.png')
          base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result', 'blended_output_0.png')
          #output_dir = os.path.join(cwd, 'data', 'temporary_result', 'right_blended_output')
          output_dir = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result')


        if i==2:
          image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask','female', 'front.png')
        #image_path = os.path.join(cwd, 'data', 'mask', 'front.png')
          image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'codeformer_output','front_00.png')
          base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result','blended_output_1.png')
          output_dir = os.path.join(cwd, 'data', 'input')



        # if i==0:

        #   image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask', 'female','right.png')
        #   image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'swapped_output','right.jpg')
        #   base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'swapped_output','front.jpg')
        #   output_dir = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result')
        


        # if i==1:
        #   image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask','female', 'left.png')
        #   image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'swapped_output','left.jpg')
        #   base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result','blended_output_0.png')
        #   output_dir = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result')


        # if i==2:
        #   image_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'mask','female', 'front.png')
        
        #   image_pathh = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'swapped_output','front.jpg')
        #   base_uv_path = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'temporary_result','blended_output_1.png')
        #   output_dir = os.path.join(cwd, 'data', 'Full_Face_Swapping', 'input')




    # Create TensorFlow graph
        graph = tf.Graph()
        with graph.as_default():
            image_tensorr = image_to_tensorr(image_path)
        
        ###defination of mask
            sess = tf.compat.v1.Session()
            with sess.as_default():
                image_arrayy = image_tensorr.eval()
        # image_tensorr = image_to_tensorr(image_path)
        # image_arrayy = image_tensorr.eval()
            image_listt = image_arrayy.tolist()
            ptt = np.squeeze(np.array(image_listt))
            uv_test_mask = np.squeeze(np.array(image_listt))
            uv_test_maskk = tf.expand_dims(uv_test_mask, axis=0) 
            uv_test_maskk = tf.expand_dims(uv_test_maskk, axis=-1)

            uv_test_maskk = tf.cast(uv_test_maskk, dtype=tf.float32)
      

     #defination of front image
        
            image_tensor = image_to_tensor(image_pathh)
            sess = tf.compat.v1.Session()
            with sess.as_default():
                image_array = image_tensor.eval()
            image_list = image_array.tolist()
            pt=np.squeeze(np.array(image_list)) 
            front_uv_batch_test = np.squeeze(np.array(image_list))
        #uv_batch_test = tf.expand_dims(front_uv_batch_test, axis=0)
            uv_batch_test = tf.expand_dims(tf.cast(front_uv_batch_test, dtype=tf.float32), axis=0)


        ###################

     ########defination of base image
            base_uv = Image.open(base_uv_path).resize((512, 512))
            base_uv = np.asarray(base_uv, np.float32) / 255
            base_uv_batch = tf.constant(base_uv[np.newaxis, ...], name="base_uv")
            base_uv_batch = tf.cast(base_uv_batch, dtype=tf.float32)
        ##########


            uv_batch = tf_blend_uv(
              base_uv_batch, uv_batch_test, uv_test_maskk, match_color=True
          )
            uv_batch = tf.identity(uv_batch, name="uv_tex")
            uv_tex_res = uv_batch[0]

        with tf.compat.v1.Session(graph=graph) as sess:
        # Initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())

        


        # Run the session and get the output
            uv_tex_res_val = sess.run(uv_tex_res)

    # Convert the NumPy array to PIL Image and save it
        uv_tex_res_image = Image.fromarray((uv_tex_res_val * 255).astype(np.uint8))
        #filename = "portrait.png"
        if i == 2:
           filename = "portrait.png"
        else:   
           filename = "blended_output_{}.png".format(i)
        
        uv_tex_res_image.save(os.path.join(output_dir, filename))

if __name__ == "__main__":
  main()
