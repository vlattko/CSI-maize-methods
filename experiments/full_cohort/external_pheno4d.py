"""Fetch explicitly selected labelled maize scans from the original public ZIP.

HTTP ranges avoid downloading tomato scans or installing a dataset framework.
Files remain external evaluation inputs, with original CRC and SHA256 evidence.
Dataset redistribution terms are not inferred from the article's CC BY licence.
"""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import struct
import urllib.request
import zipfile
import zlib

URL='https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip'


class RangeArchive(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.position=0
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as response:
            self.size=int(response.headers['Content-Length']);self.etag=response.headers.get('ETag')
            self.last_modified=response.headers.get('Last-Modified')
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative archive seek')
        return self.position
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        if not size:return b''
        size=min(size,self.size-self.position)
        if size<=0:return b''
        start=self.position;end=start+size-1
        headers={'Range':f'bytes={start}-{end}'}
        if self.etag:headers['If-Range']=self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url,headers=headers),timeout=90) as response:
            if response.status!=206:raise ValueError('Server did not honour a bounded range request')
            if self.etag and response.headers.get('ETag')!=self.etag:raise ValueError('Archive changed during retrieval')
            data=response.read(size+1)
        if len(data)!=size:raise ValueError('Archive range length mismatch')
        self.position+=size;return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--download',choices=['none','last_two_per_plant','all_labelled'],default='none')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    remote=RangeArchive(URL)
    with zipfile.ZipFile(remote) as archive:infos=archive.infolist()
    maize=[i for i in infos if 'Maize' in i.filename and i.filename.endswith('_a.txt') and not i.is_dir()]
    if not maize:
        print(json.dumps([i.filename for i in infos[:30]],indent=2));raise ValueError('No labelled maize scans in expected archive schema')
    catalog=[dict(path=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,crc32=i.CRC,
                  compression=i.compress_type,header_offset=i.header_offset) for i in maize]
    (out/'catalog.json').write_text(json.dumps(dict(source_url=URL,archive_bytes=remote.size,etag=remote.etag,
        last_modified=remote.last_modified,labelled_maize_files=catalog,
        redistribution_status='Not inferred; external inputs are not bundled with original study data.'),indent=2),encoding='utf-8')
    print(json.dumps({'labelled_maize_files':len(maize),'compressed_total':sum(i.compress_size for i in maize),'examples':[i.filename for i in maize[:5]]}),flush=True)
    if args.download=='none':return
    if args.download=='last_two_per_plant':
        groups=defaultdict(list)
        for item in maize:groups[PurePosixPath(item.filename).parent.as_posix()].append(item)
        chosen=[i for group in sorted(groups) for i in sorted(groups[group],key=lambda i:i.filename)[-2:]]
    else:chosen=sorted(maize,key=lambda i:i.filename)
    manifest=[]
    for info in chosen:
        relative=PurePosixPath(info.filename)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe external filename')
        target=out/'original'/Path(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            data=target.read_bytes()
        else:
            remote.seek(info.header_offset);header=remote.read(30)
            values=struct.unpack('<4s5H3I2H',header)
            if values[0]!=b'PK\x03\x04' or values[2]&1:raise ValueError('Unsupported local archive header')
            remote.seek(info.header_offset+30+values[-2]+values[-1]);compressed=remote.read(info.compress_size)
            if info.compress_type==zipfile.ZIP_DEFLATED:data=zlib.decompress(compressed,-15)
            elif info.compress_type==zipfile.ZIP_STORED:data=compressed
            else:raise ValueError('Unsupported archive compression')
            if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:raise ValueError('External scan CRC/size mismatch')
            temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(target)
        if len(data)!=info.file_size or zlib.crc32(data)&0xffffffff!=info.CRC:raise ValueError('Existing external input differs')
        manifest.append(dict(path=target.relative_to(out).as_posix(),source_member=info.filename,
                             bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=info.CRC))
        (out/'download_manifest.json').write_text(json.dumps(dict(status='running',source_url=URL,
            selection=args.download,files=manifest),indent=2),encoding='utf-8')
        print(json.dumps({'downloaded':len(manifest),'planned':len(chosen),'member':info.filename,'bytes':len(data)}),flush=True)
    (out/'download_manifest.json').write_text(json.dumps(dict(status='completed',source_url=URL,
        selection=args.download,files=manifest),indent=2),encoding='utf-8')


if __name__=='__main__':main()
