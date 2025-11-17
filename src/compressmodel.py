import timeit
from imclaslib.config import Config
import torch
import torchvision.transforms as transforms
import json
from imclaslib.dataset import datasetutils
from imclaslib.files import modelloadingutils, pathutils
from imclaslib.logging.loggerfactory import LoggerFactory
from imclaslib.models import modelfactory
#from torch.export import Dim
from torch.cuda.amp import autocast

config = Config("default_config.yml")
logger = LoggerFactory.setup_logging("logger", config, log_file=pathutils.combine_path(config, 
    pathutils.get_log_dir_path(config), 
    f"{config.model_name}_{config.model_image_size}_{config.model_weights}",
    f"train__{pathutils.get_datetime()}.log"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    model = modelfactory.create_model(config).to(device)
    # Load the existing model
    modelData = modelloadingutils.load_model(pathutils.get_model_to_load_path(config), config)
    model.load_state_dict(modelData['model_state_dict'])

    # Switch the model to evaluation mode
    model.eval()
    test_loader = datasetutils.get_data_loader_by_name("test", config=config, num_workers=0, device=device)
    dataiter = iter(test_loader)
    dataiter = next(dataiter)
    images = dataiter['image'].to(device)
    images = images.half()
    model = model.to(memory_format=torch.channels_last)

    # batch = Dim("batch")
    # dynamic_shapes = {"x": {0: batch}}
    # exported_program = torch.export.export(model, (images,), dynamic_shapes=dynamic_shapes)

    # torch.export.save(exported_program, 'torch_export_test.pt2')

    # input = torch_tensorrt.Input(
    #     min_shape=(1, 3, config.model_image_size, config.model_image_size),
    #     opt_shape=(16, 3, config.model_image_size, config.model_image_size),
    #     max_shape=(16, 3, config.model_image_size, config.model_image_size),
    #     dtype=torch.half, name="x")
    # input_shape = (1, 3, config.model_image_size, config.model_image_size)
    # dynamic_axes = {'x' : {0 : 'batch_size'}}
    # input = torch_tensorrt.Input(shape=input_shape, dtype=torch.half, dynamic_axes=dynamic_axes, name="x")
    #with autocast(enabled=True):
    # spec = {
    #     "forward": torch_tensorrt.ts.TensorRTCompileSpec(
    #         **{
    #             "inputs": [torch_tensorrt.Input([8, 3, 512, 512])],
    #             "enabled_precisions": {torch.float, torch.half},
    #             "refit": False,
    #             "debug": False,
    #             "device": {
    #                 "device_type": torch_tensorrt.DeviceType.GPU,
    #                 "gpu_id": 0,
    #                 "dla_core": 0,
    #                 "allow_gpu_fallback": True,
    #             },
    #             "num_avg_timing_iters": 1,
    #         }
    #     )
    # }

    with autocast(enabled=True):
        model(images)
        model(images)
        execution_time = timeit.timeit('model(images)', globals=globals(), number=2)
        print(f'Average {execution_time/2} per batch')
        with torch.jit.optimized_execution(True):
            scripted_model = torch.jit.script(model, images)
            scripted_model = torch.jit.freeze(scripted_model)
            #trt_model = torch._C._jit_to_backend("tensorrt", scripted_model, spec)
            #trt_model.forward(images)
            #trt_model.forward(images)
            execution_time = timeit.timeit('scripted_model(images)', globals=globals(), number=2)
            print(f'Average {execution_time/2} per batch')
            torch.jit.save(scripted_model, "bright_sound.pt")

    # model = model.half()
    # exported_program: torch.export.ExportedProgram = torch.export.export(
    #     model, args=(images,)
    # )  

    # exported_program.module()(images)
    # exported_program.module()(images)
    # execution_time = timeit.timeit('exported_program.module()(images)', globals=globals(), number=20)
    # print(f'Average {execution_time/10} per batch')
    #    #trt_gm = torch_tensorrt.compile(model, ir="dynamo", inputs=[torch_tensorrt.Input(name="x", shape=images.shape, dtype=torch.half)], enabled_precisions = {torch.half, torch.float})
    # torch.export.save(exported_program, "trt_model.ep")
    # with autocast(enabled=True):
    #     batch = Dim("batch") 
    #     dynamic_shapes = {"x": {0: batch}}     
    #     exported_program: torch.export.ExportedProgram = export(
    #         model, args=(images,), dynamic_shapes=dynamic_shapes
    #     )  
    #     torch.export.save(exported_program, "trt_model.ep")
    # onnx_program = torch.onnx.export(model,         # model being run 
    #      images,       # model input (or a tuple for multiple inputs) 
    #      "onnx_model.onnx",       # where to save the model  
    #      export_params=True,  # store the trained parameter weights inside the model file 
    #      opset_version=17,    # the ONNX version to export the model to 
    #      do_constant_folding=True,  # whether to execute constant folding for optimization 
    #      input_names = ['modelInput'],   # the model's input names 
    #      output_names = ['modelOutput']) # the model's output names 
         #dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                #'modelOutput' : {0 : 'batch_size'}}) 
    #onnx_program = torch.onnx.dynamo_export(model, images)
    #onnx_program.save("onnx_model.onnx")
    #onnx_model = onnx.load("onnx_model.onnx")
    #onnx.checker.check_model(onnx_model)
    #torch.jit.save(trt_gm, "trt_model.ts")
