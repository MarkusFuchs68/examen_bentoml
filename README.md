## Examen BentoML Markus Fuchs

Folder structure: 

```bash       
├── examen_bentoml          
│   ├── data       
│   │   ├── processed      
│   │   └── raw           
│   ├── models      
│   ├── src       
│   ├── bentofile.yaml
│   ├── bento_image.tar
│   └── README.md
```

## Commands

##### Build the Bento

```bash
bentoml build
```
##### Serve the Bento

```bash
cd src
bentoml serve service:rf_service
```

##### Load the docker image into the local docker daemon

```bash
docker load -i bento_image.tar
```

##### Run the docker container

```bash
docker run --rm -p 3000:3000 admission_rf_service:vmvkd4rnbwbdxjgr
```

##### Run unit tests (start the docker container first)

```bash
pytest -v --disable-warnings
```
