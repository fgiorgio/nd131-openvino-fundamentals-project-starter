# Project Write-Up

## Explaining Custom Layers

The process of conversion from the supported frameworks to the Inference Engine formats is automatic for topologies with the standard layers that are known to the Model Optimizer tool.

The Inference Engine loads the layers from the input model IR files into the specified device plugin, which will search a list of known layer implementations for the device. If your topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error.

The main purpose of registering a custom layer within the Model Optimizer is to define the shape inference (how the output shape size is calculated from the input size). Once the shape inference is defined, the Model Optimizer does not need to call the specific training framework again.

Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device.

The following is an example of conversion using an extension to support custom layers:

python mo_tf.py --input_model=frozen_inference_graph.pb --tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config=pipeline.config --reverse_input_channels

## Comparing Model Performance

Cloud infrastructure takes time for data to travel from the edge device back to the center for processing. Network latency can have serious consequences for IoT devices. Edge computing offers a solution to the latency problem by relocating crucial data processing to the edge of the network. For many companies, the cost savings alone can be a driver towards deploying an edge-computing architecture. Companies that embraced the cloud for many of their applications may have discovered that the costs in bandwidth were higher than they expected.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

- Customers counter in a store
- Security camera into apartments
- People counter on airplanes, trains or ships
- Queue management systems

Each of these use cases would be useful because it would help checking for number of people automatically, improving processes and security.

It is possible to calculate the overall conversion rate and also identify the particular category of merchandise that is bringing in the sales of a store.

Many airports rightfully leverage the benefits of people counting technologies to create a positive passenger experience.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

- Only when it is possible to view the specific features with sufficient contrast is it possible to evaluate them using image processing software. This is generally achieved by illuminating the object using a light source. The principle of illuminating the object may seem banal, but experience shows that one of the main difficulties in image processing is to make object visible to the camera.
- Accuracy rates for object identification and classification have gone from 50 percent to 99 percent in less than a decade and today’s systems are more accurate than humans at quickly detecting and reacting to visual inputs. The better a model can generalize to ‘unseen’ data, the better predictions and insights it can produce, which in turn deliver more business value. The cost of errors can be huge, but optimizing model accuracy mitigates that cost.  The benefits of improving model accuracy help avoid considerable time, money, and undue stress.
- High-quality images are fundamental for image processing. The difference in image quality directly affects the inspection accuracy when using image processing technology. Camera selection according to the application is also important. Scenes captured with short-focal-length lenses appear expanded in depth, while those captured with long lenses appear compressed. Higher quality image give the model more information but require more neural network nodes and more computing power to process.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD MobileNet v1 COCO
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments: 
  -- input_model=/home/workspace/model/SSD_MobileNet_v1_COCO/frozen_inference_graph.pb
  -- tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  -- tensorflow_object_detection_api_pipeline_config=/home/workspace/model/SSD_MobileNet_v1_COCO/pipeline.config
  -- reverse_input_channels
  - The model was insufficient for the app because there were inaccurate detections for too much frames
  - I tried to improve the model for the app by addind a custom algorithm to ignore a specific number of frames after detection state changes and changing probability threshold parameter
  
- Model 2: SSD MobileNet v2 COCO
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments: 
  -- input_model=/home/workspace/model/SSD_MobileNet_v2_COCO/frozen_inference_graph.pb
  -- tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  -- tensorflow_object_detection_api_pipeline_config=/home/workspace/model/SSD_MobileNet_v2_COCO/pipeline.config
  -- reverse_input_channels
  - The model was insufficient for the app because there were inaccurate detections for too much frames
  - I tried to improve the model for the app by addind a custom algorithm to ignore a specific number of frames after detection state changes and changing probability threshold parameter

- Model 3: SSD Inception v2 COCO
  - http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments: 
  -- input_model=/home/workspace/model/SSD_Inception_v2_COCO/frozen_inference_graph.pb
  -- tensorflow_use_custom_operations_config=/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  -- tensorflow_object_detection_api_pipeline_config=/home/workspace/model/SSD_Inception_v2_COCO/pipeline.config
  -- reverse_input_channels
  - The model was insufficient for the app because there were inaccurate detections for too much frames and inference was too slow
  - I tried to improve the model for the app by addind a custom algorithm to ignore a specific number of frames after detection state changes and changing probability threshold parameter
