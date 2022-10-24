import os
import requests
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
import shutil
import zipfile

from shapely.geometry import shape

import rectifiedgrid as rg
import matplotlib.pyplot as plt
import coreapi
import numpy as np
import logging
import subprocess
import cartopy.io.img_tiles as cimgt
from urllib.parse import urljoin
from rasterio.enums import MergeAlg
import glob

logger = logging.getLogger('tools4msp_apiclient')


class Tools4MSPApiCLient(object):
    def __init__(self, APIURL, TOKEN):
        auth = coreapi.auth.TokenAuthentication(
            scheme='Token',
            token=TOKEN
        )
        self.client = coreapi.Client(auth=auth)
        self.schema = self.client.get(APIURL)
        self._coded_labels = None
        self._domain_areas = None
        self.APIURL = APIURL
        self.TOKEN = TOKEN

    @property
    def domain_areas(self):
        if self._domain_areas is None:
            domain_areas = self.client.action(self.schema, ['api', 'domainareas', 'list'])
            domain_areas_list = {}
            for i in domain_areas:
                domain_areas_list[i['label']] = i['url']
            self._domain_areas = domain_areas_list
        return self._domain_areas

    def get_domain_area(self, label):
        domain_areas = self.domain_areas
        url = domain_areas.get(label)
        if url is None:
            return None
        da = self.client.get(url)
        da['geo'] = shape(da['geo'])
        return da


    @property
    def coded_labels(self):
        if self._coded_labels is None:
            coded_labels = self.client.action(self.schema, ['api', 'codedlabels', 'list'])
            self._coded_labels = {i['code']: i for i in coded_labels}
        return {k: i['url'] for k, i in self._coded_labels.items()}

    def get_coded_label(self, code):
        self.coded_labels
        return self._coded_labels.get(code, None)

    def get_run(self, runid):
        params = {'id': runid}
        run =  self.client.action(self.schema, ['api', 'casestudyruns', 'read'],
                       params=params)
        return run

    def get_cs(self, csid):
        params = {'id': csid}
        cs =  self.client.action(self.schema, ['api', 'casestudies', 'read'],
                       params=params)
        return cs

    def get_output(self, runid, code):
        run = self.get_run(runid)
        coded_label = self.coded_labels.get(code)
        for o in run['outputs']:
            if o['coded_label'] == coded_label:
                return o

    def get_outputlayer(self, runid, code):
        run = self.get_run(runid)
        coded_label = self.coded_labels.get(code)
        for o in run['outputlayers']:
            if o['coded_label'] == coded_label:
                return rg.read_raster(o['file'])

    def get_layer(self, csid, code):
        cs = self.get_cs(csid)
        coded_label = self.coded_labels.get(code)
        for o in cs['layers']:
            if o['coded_label'] == coded_label:
                return rg.read_raster(o['file'])

    def create_and_upload(self, otype, parent_id, clurl, filepath=None,
                          thumbnailpath=None, replace=True):
        params = {
            'parent_lookup_casestudy__id': parent_id,
        }
        olist = self.client.action(self.schema,
                              ['api', 'casestudies', otype, 'list'],
                              params=params)

        created_obj = None
        for o in olist:
            if o['coded_label'] == clurl:
                created_obj = o

        params = {
            'parent_lookup_casestudy__id': parent_id,
            'coded_label': clurl,
        }

        if not replace or created_obj is None:
            created_obj = self.client.action(self.schema, ['api', 'casestudies', otype, 'create'], params=params)

        # upload file
        if filepath is not None:
            url = created_obj['url'] + 'upload/'
            input_file = filepath
            with open(input_file, 'rb') as f:
                files = {'file': f}
                headers = {'Authorization': 'Token {}'.format(self.TOKEN)}
                r = requests.put(url, headers=headers, files=files)
                if r.status_code != 201:
                    print(r.status_code, r.content)

        # upload thumbnail
        if thumbnailpath is not None:
            url = created_obj['url'] + 'tupload/'
            input_file = thumbnailpath
            with open(input_file, 'rb') as f:
                files = {'file': f}
                headers = {'Authorization': 'Token {}'.format(self.TOKEN)}
                r = requests.put(url, headers=headers, files=files)
                if r.status_code != 201:
                    print(r.status_code, r.content)

class GeoNode(object):
    def __init__(self, url, username, password):
        self.url = url
        self.login_url = os.path.join(self.url, 'account', 'login/')
        self.username = username
        self.password = password
        self._layers = None
        self.client = None
        self.login()

    def login(self):
        client = requests.session()

        # Retrieve the CSRF token first
        client.get(self.login_url)  # sets cookie
        # print(client.cookies)
        csrftoken = client.cookies.get('csrftoken')

        login_data = dict(login=self.username,
                          password=self.password,
                          csrfmiddlewaretoken=csrftoken,
                          next='/')
        r = client.post(self.login_url,
                        data=login_data,
                        headers=dict(Referer=self.login_url))
        self.client = client

    @property
    def layers(self):
        if self._layers is None:
            next_url = '/api/layers/'
            layers = []
            while next_url is not None:
                print("Loading ...", next_url)
                url = urljoin(self.url, next_url)
                print("\t", url)
                response = self.client.get(url, params={'limit': 200})
                r_json = response.json()
                layers += r_json['objects']
                next_url = r_json['meta']['next']
            self._layers = layers
        return self._layers

    def get_layer(self, name):
        """This is deprecated because name is not unique. Use get_layer_by_id instead"""
        for l in self.layers:
            if l['name'] == name:
                return l
        return None

    def get_layer_by_id(self, id):
        for l in self.layers:
            if l['id'] == id:
                return l
        return None

    def download_layer(self, name, fpath):
        l = self.get_layer(name)
        durl = os.path.join(self.url, 'download', str(l['id']))
        print(durl)

        r = self.client.get(durl, stream=True)
        print(durl, r)
        if r.status_code == 200:
            with open(fpath, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        else:
            print("Download error")

    def md_extra(self, id, fields=None):
        l = self.get_layer_by_id(id)
        url = urljoin(self.url, '/layers/{}/metadata_detail'.format(l['alternate']))
        req = self.client.get(url)
        if req.status_code != 200:
            print(req.status_code)
            return None
        soup = BeautifulSoup(req.content, 'html.parser')
        article = {}
        article_fields = {}
        _article = soup.find('article')
        key = None
        for c in _article.contents:
            if c.name == 'span':
                key = c.string
                article[key] = {}
            if c.name == 'dl':
                for dt in c.find_all('dt'):
                    field = dt.string
                    value = ' '.join(dt.find_next_sibling('dd').stripped_strings)
                    article[key][field] = value
                    if fields is not None and field in fields:
                        article_fields[field] = value
                    # print("TAG", key, dt.string, article[key][dt.string])
        if fields is not None:
            return article_fields
        return article


class GeoDataBuilder(object):
    api = None
    source = None

    def __init__(self, APIURL, TOKEN, workdir="/tmp/", username=None, password=None):
        self.tclient = Tools4MSPApiCLient(APIURL, TOKEN)
        self.workdir = workdir
        self.downloaded = os.path.join(self.workdir, 'downloaded')
        self.inputs = os.path.join(self.workdir, 'inputs')
        self.source = GeoNode('https://www.portodimare.eu/',
                    username=username, password=password)


    def download_remote(self, name):
        zpath = os.path.join(self.downloaded, '{}.zip'.format(name))
        unzipdir = os.path.join(self.downloaded, '{}'.format(name))

        if not os.path.isdir(unzipdir):
            if not os.path.isfile(zpath):
                logger.debug("Downloading {}".format(name))
                self.source.download_layer(name, zpath)
            with zipfile.ZipFile(zpath, 'r') as zip_ref:
                logger.debug("Extracting {}".format(name))
                zip_ref.extractall(unzipdir)

    def transform(self, name, vect=True):
        unzipdir = os.path.join(self.downloaded, '{}'.format(name))
        unzipdir_reprojected = os.path.join(self.downloaded, '{}_reprojected'.format(name))

        if vect:
            options = ['/usr/bin/ogr2ogr',
                       '-overwrite',
                       '-t_srs', 'epsg:3035',
                       unzipdir_reprojected,
                       unzipdir
                       ]
        else:
            rfpath = glob.glob("{}/*.tif".format(unzipdir))[0]
            # rfpath = '{}/{}.tif'.format(unzipdir, name)
            rfpath_reprojected = '{}/{}.tif'.format(unzipdir_reprojected, name)
            if not os.path.exists(unzipdir_reprojected):
                os.makedirs(unzipdir_reprojected)
            options = ['/usr/bin/gdalwarp',
                       '-overwrite',
                       '-t_srs', 'epsg:3035',
                       rfpath,
                       rfpath_reprojected
                       ]

        subprocess.check_call(options, stderr=subprocess.STDOUT)

    def get_remote(self, name, resolution, grid=None, column=None, query=None,
                   transform='scale', fillvalue=0, merge_alg=MergeAlg.replace,
                   store_type=None, path_only=False):
        unzipdir = os.path.join(self.downloaded, '{}'.format(name))
        unzipdir_reprojected = os.path.join(self.downloaded, '{}_reprojected'.format(name))

        self.download_remote(name)
        if store_type is None:
            l = self.source.get_layer(name)
            store_type = l['store_type']

        if store_type == 'dataStore':
            fpath = '{}'.format(unzipdir_reprojected)
        elif store_type == 'coverageStore':
            fpath = '{}/{}.tif'.format(unzipdir_reprojected, name)
        else:
            raise Exception("Unrecognized data store {}".format(store_type))

        self.transform(name, vect=store_type == 'dataStore')

        if path_only:
            return fpath
        return self.rasterize_file(fpath, store_type, resolution, grid, column, query, fillvalue,
                             merge_alg)

    def rasterize_file(self, fpath, store_type, resolution, grid=None, column=None, query=None,
                 fillvalue=0., merge_alg=MergeAlg.replace):
        if grid is None:
            raster = rg.read_vector(fpath, res=resolution,
                                    rounded_bounds=True, epsg=3035, 
                                    query=query, fillvalue=np.nan)
            # raster = np.ma.masked_where(raster==0, raster)
            # raster.fill_underlying_data(raster.fill_value)
        elif store_type == 'dataStore':
            raster = rg.read_vector(fpath, res=resolution, rounded_bounds=True,
                                    grid=grid,
                                    column=column, epsg=3035, query=query,
                                    fillvalue=fillvalue,
                                    merge_alg=merge_alg)
            raster = raster.where(raster > 0, 0).where(~grid.isnull()).rio.write_nodata(np.nan)
            # raster.mask = grid.mask.copy()
        elif store_type == 'coverageStore':
            raster = rg.read_raster(fpath, grid=grid)
            raster = raster.where(raster>0, 0).where(~grid.isnull()).rio.write_nodata(np.nan)
            # raster = raster.to_srs_like(grid)
            # TODO
            # raster.
            # raster.mask = grid.mask.copy()
            # raster[raster == raster.fill_value] = 0
        return raster


    def upload_layer(self, parent_ids, raster, code, logcolor=False,
                     legend=False, zoomlevel=9, alpha=None, figsize=None,
                     stamen=True, dry_run=True): #ATTENZIONE: e' stato aggiunto perché di default non sincronizzi
        #
        if dry_run:
            return raster
        rpath = os.path.join(self.inputs, '{}.geotiff'.format(code))
        thumbrpath = os.path.join(self.inputs, '{}.png'.format(code))

        raster.rio.to_raster(rpath, driver="GTiff") # , nodata=raster.fill_value)
        if figsize is not None:
            plt.figure(figsize=figsize)
        ax, mapimg = raster.rg.plotmap(logcolor=logcolor, legend=legend, alpha=alpha)
        if stamen:
            ax.add_image(cimgt.Stamen('toner-lite'), zoomlevel)
        plt.savefig(thumbrpath)
        plt.show()
        # upload grid
        clurl = None
        coded_labels = self.tclient.client.action(self.tclient.schema, ['api', 'codedlabels', 'list'])
        coded_labels_list = {}
        for i in coded_labels:
            coded_labels_list[i['code']] = i['url']
        clurl = coded_labels_list.get(code)
        if clurl is None:
            print('Invalid code', code)
            return None
        for parent_id in parent_ids:
            print("Uploading", code, "on cs", parent_id)
            self.tclient.create_and_upload('layers', parent_id, clurl, rpath, thumbrpath)
        return raster

# DEPRECATED: user get_remote + upload_layer
def flat_upload(name, code, grid=None):
    zpath = 'downloaded/{}.zip'.format(name)
    unzipdir = 'downloaded/{}'.format(name)
    vfpath = '{}/{}.shp'.format(unzipdir, name)
    rfpath = '{}/{}.tif'.format(unzipdir, name)
    #
    rpath = 'inputs/{}.geotiff'.format(code)
    thumbrpath = 'inputs/{}.png'.format(code)

    l = g.get_layer(name)
    if not os.path.isfile(zpath):
        g.download_layer(name, zpath)
    with zipfile.ZipFile(zpath, 'r') as zip_ref:
        zip_ref.extractall(unzipdir)
    if grid is None:
        raster = rg.read_vector(vfpath, res=resolution, eea=True)
        raster = np.ma.masked_where(raster==0, raster)
        raster.write_raster(rpath)
    elif l['store_type'] == 'dataStore':
        raster = rg.read_vector(vfpath, res=resolution, eea=True, grid=grid)
        raster.mask = grid.mask.copy()
        raster.write_raster(rpath)
    elif l['store_type'] == 'coverageStore':
        raster = rg.read_raster(rfpath)
        raster = raster.to_srs_like(grid)
        raster.mask = grid.mask.copy()
        raster[raster == raster.fill_value] = 0
        raster.write_raster(rpath, nodata=raster.fill_value)
    raster.plotmap()
    plt.savefig(thumbrpath)
    # upload grid
    clurl = client.action(schema, ['api', 'codedlabels', 'read'],
                          params={'code': code})['url']
    create_and_upload('layers', parent_id, clurl, rpath, thumbrpath)
    return raster
