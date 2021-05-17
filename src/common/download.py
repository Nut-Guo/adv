import os
import requests
import logging
import tqdm as tq
import zipfile
import tarfile


def download_file(url: str, path: str, verbose: bool = False) -> None:
    """
    Download file with progressbar

    Usage:
        download_file('http://web4host.net/5MB.zip')
    """
    if not os.path.exists(path):
        os.makedirs(path)
    local_filename = os.path.join(path, url.split('/')[-1])

    if not os.path.exists(local_filename):
        r = requests.get(url, stream=True)
        file_size = int(r.headers.get('Content-Length', 0))
        chunk = 1
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)
        if verbose:
            logging.info(f'file size: {file_size}\n# bars: {num_bars}')
        with open(local_filename, 'wb') as fp:
            for chunk in tq(
                r.iter_content(chunk_size=chunk_size),
                total=num_bars,
                unit='KB',
                desc=local_filename,
                leave=True  # progressbar stays
            ):
                fp.write(chunk)  # type: ignore

    if '.zip' in local_filename:
        if os.path.exists(local_filename):
            with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                zip_ref.extractall(path)
    elif '.tar.gz' in local_filename:
        if os.path.exists(local_filename):
            with tarfile.open(local_filename, 'r') as tar_ref:
                tar_ref.extractall(path)


def download_data(url: str, path: str = "data/") -> None:
    """
    Downloads data automatically from the given url to the path. Defaults to data/ for the path.
    Automatically handles .csv, .zip

    Example::

        from flash import download_data

    Args:
        url: path
        path: local

    """
    download_file(url, path)


def main():
    pass


if __name__ == '__main__':
    main()