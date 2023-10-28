
import requests
import json
import io
import base64
import os

from PIL import Image

# from math import floor
# def clamp( v : float | int, low=-9e99, high=9e99 ) -> float | int:
# 	return max( low, min( v, high ) )

def get_sys_info( ) -> dict:
	data = requests.get(f'http://127.0.0.1:7860/internal/sysinfo').json()
	return {
		"Path" : data['Data path'],
		"Extensions" : [ ext['name'] for ext in data['Extensions'] ],
		"RAM" : data['RAM'],
		"CPU" : data["CPU"],
		"CmdLineArgs" : data['Commandline'][1:]
	}
SYS_INFO = get_sys_info()

def get_stable_diffusion_options( ) -> dict:
	return requests.get(f'http://127.0.0.1:7860/sdapi/v1/options').json()

def set_stable_diffusion_options( options : dict ) -> None:
	return requests.post(f'http://127.0.0.1:7860/sdapi/v1/options', json=options).json()

def refresh_checkpoints( ) -> None:
	requests.post(f'http://127.0.0.1:7860/sdapi/v1/refresh-checkpoints')

def refresh_vae( ) -> None:
	requests.post(f'http://127.0.0.1:7860/sdapi/v1/refresh-vae')

def refresh_loras( ) -> None:
	requests.post(f'http://127.0.0.1:7860/sdapi/v1/refresh-loras')

def refresh_hypernetworks( ) -> None:
	refresh_loras( )

def refresh_embeddings( ) -> None:
	refresh_loras( )

def get_hypernetworks( ) -> list:
	refresh_hypernetworks( )
	hypernetworks = requests.get(f'http://127.0.0.1:7860/sdapi/v1/hypernetworks').json()
	for network in hypernetworks:
		network['path'] = network.pop('path')[len(SYS_INFO['Path'])+1:]
	return hypernetworks

def get_embeddings( ) -> list:
	refresh_embeddings( )
	return list(requests.get(f'http://127.0.0.1:7860/sdapi/v1/embeddings').json().keys())

def get_vaes( ) -> list:
	refresh_vae( )
	vaes = requests.get(f'http://127.0.0.1:7860/sdapi/v1/sd-vae').json()
	for vae in vaes:
		vae['name'] = vae.pop('model_name')
		vae['path'] = vae.pop('filename')[len(SYS_INFO['Path'])+1:]
	return vaes

def get_models( ) -> list:
	refresh_checkpoints()
	return [ {
		"title" : model['title'],
		"model_name" : model['model_name'],
		"size_bytes" : os.path.getsize( model['filename'] )
	} for model in requests.get(
		f'http://127.0.0.1:7860/sdapi/v1/sd-models'
	).json() ]

def get_loras( ) -> list:
	refresh_loras( )
	loras = requests.get(f'http://127.0.0.1:7860/sdapi/v1/loras').json()
	for lora in loras:
		lora['path'] = lora['path'][len(SYS_INFO['Path'])+1:]
		meta = lora.pop('metadata')
		if len(list(meta.keys())) > 0:
			lora['type'] = meta['ss_network_module'].split('.')[-1]
		else:
			lora['type'] = 'unknown'
	return loras

def get_samplers( ) -> list:
	return [ item['name'] for item in requests.get(f'http://127.0.0.1:7860/sdapi/v1/samplers').json() ]

def get_upscalers( ) -> list:
	return [ item['name'] for item in requests.get(f'http://127.0.0.1:7860/sdapi/v1/upscalers').json() ]

def reload_checkpoint( ) -> None:
	requests.post(f'http://127.0.0.1:7860/sdapi/v1/reload-checkpoint')

def unload_checkpoint( ) -> None:
	requests.post(f'http://127.0.0.1:7860/sdapi/v1/unload-checkpoint')

def get_txt2img_scripts( ) -> list:
	return requests.get(f'http://127.0.0.1:7860/sdapi/v1/scripts')['txt2img']

def set_prompt_options( options : dict ) -> None:
	return requests.post('http://127.0.0.1:7860/sdapi/v1/options', json=options)

def get_prompt_options( ) -> dict:
	return requests.get('http://127.0.0.1:7860/sdapi/v1/options')

def update_prompt_options( options : dict ) -> None:
	values = get_prompt_options( )
	values.update(options)
	set_prompt_options( values )

def txt2img(
	checkpoint : str = "v1-5-pruned-emaonly.safetensors [6ce0161689]",
	checkpoint_vae : str = None,
	# embeddings : list[tuple] = None,
	# hypernetworks : list[tuple] = None,
	# loras : list[tuple] = None,

	prompt : str = "",
	negative_prompt : str = "",

	steps : int = 25,
	cfg_scale : int = 7,
	sampler_name : str = "Euler a",
	width : str = 512,
	height : str = 512,

	batch_size : int = 1,
	batch_count : int = 1,
	seed : int = -1,
) -> dict:

	# change models / vae
	update_prompt_options({
		'sd_model_checkpoint': checkpoint,
		'sd_vae' : checkpoint_vae
	})

	# run txt2img
	return requests.post(f"http://127.0.0.1:7860/sdapi/v1/txt2img", json={
		"prompt" : prompt,
		"negative_prompt" : negative_prompt,
		"steps" : steps,
		"seed" : seed,
		"sampler_name" : sampler_name,
		"batch_size" : batch_size,
		"n_iter" : batch_count,
		"cfg_scale" : cfg_scale,
		"width" : width,
		"height" : height
	})

# TODO:
# http://127.0.0.1:7860/docs#/default/text2imgapi_sdapi_v1_txt2img_post
# /sdapi/v1/png-info
# /sdapi/v1/progress
# /sdapi/v1/interrupt
# /sdapi/v1/skip
# /sdapi/v1/txt2img
# /queue/status
# /internal/ping

# <lora:NAME:WEIGHT>
# <hypernetwork:NAME:WEIGHT>

# if __name__ == '__main__':

	# 1
	# data = get_models()
	# with open("info.json", "w") as file:
	# 	file.write( json.dumps(data, indent=4) )

	# 2
	# data = txt2img( "pug", steps=25 )
	# with open("info.json", "w") as file:
	# 	file.write( json.dumps(data, indent=4) )
	# for index, img in enumerate( data['images'] ):
	# 	image = Image.open(io.BytesIO(base64.b64decode(data['images'][0])))
	# 	image.save(f'{index}_output.png')
