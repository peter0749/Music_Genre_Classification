import dotenv
import pydot
import requests
import numpy as np
import pandas as pd
import ctypes
import shutil
import multiprocessing
import multiprocessing.sharedctypes as sharedctypes
import os.path
import ast


# Number of samples per 30s audio clip.
# TODO: fix dataset to be constant.
NB_AUDIO_SAMPLES = 1321967
SAMPLING_RATE = 44100

# Load the environment from the .env file.
dotenv.load_dotenv(dotenv.find_dotenv())


class FreeMusicArchive:

    BASE_URL = 'https://freemusicarchive.org/api/get/'

    def __init__(self, api_key):
        self.api_key = api_key

    def get_recent_tracks(self):
        URL = 'https://freemusicarchive.org/recent.json'
        r = requests.get(URL)
        r.raise_for_status()
        tracks = []
        artists = []
        date_created = []
        for track in r.json()['aTracks']:
            tracks.append(track['track_id'])
            artists.append(track['artist_name'])
            date_created.append(track['track_date_created'])
        return tracks, artists, date_created

    def _get_data(self, dataset, fma_id, fields=None):
        url = self.BASE_URL + dataset + 's.json?'
        url += dataset + '_id=' + str(fma_id) + '&api_key=' + self.api_key
        r = requests.get(url)
        r.raise_for_status()
        if r.json()['errors']:
            raise Exception(r.json()['errors'])
        data = r.json()['dataset'][0]
        r_id = data[dataset + '_id']
        if r_id != str(fma_id):
            raise Exception('The received id {} does not correspond to'
                            'the requested one {}'.format(r_id, fma_id))
        if fields is None:
            return data
        if type(fields) is list:
            ret = {}
            for field in fields:
                ret[field] = data[field]
            return ret
        else:
            return data[fields]

    def get_track(self, track_id, fields=None):
        return self._get_data('track', track_id, fields)

    def get_album(self, album_id, fields=None):
        return self._get_data('album', album_id, fields)

    def get_artist(self, artist_id, fields=None):
        return self._get_data('artist', artist_id, fields)

    def get_all(self, dataset, id_range):
        index = dataset + '_id'

        id_ = 2 if dataset is 'track' else 1
        row = self._get_data(dataset, id_)
        df = pd.DataFrame(columns=row.keys())
        df.set_index(index, inplace=True)

        not_found_ids = []

        for id_ in id_range:
            try:
                row = self._get_data(dataset, id_)
            except:
                not_found_ids.append(id_)
                continue
            row.pop(index)
            df.loc[id_] = row

        return df, not_found_ids

    def download_track(self, track_file, path):
        url = 'https://files.freemusicarchive.org/' + track_file
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    def get_track_genres(self, track_id):
        genres = self.get_track(track_id, 'track_genres')
        genre_ids = []
        genre_titles = []
        for genre in genres:
            genre_ids.append(genre['genre_id'])
            genre_titles.append(genre['genre_title'])
        return genre_ids, genre_titles

    def get_all_genres(self):
        df = pd.DataFrame(columns=['genre_parent_id', 'genre_title',
                                   'genre_handle', 'genre_color'])
        df.index.rename('genre_id', inplace=True)

        page = 1
        while True:
            url = self.BASE_URL + 'genres.json?limit=50'
            url += '&page={}&api_key={}'.format(page, self.api_key)
            r = requests.get(url)
            for genre in r.json()['dataset']:
                genre_id = int(genre.pop(df.index.name))
                df.loc[genre_id] = genre
            assert (r.json()['page'] == str(page))
            page += 1
            if page > r.json()['total_pages']:
                break

        return df


class Genres:

    def __init__(self, genres_df):
        self.df = genres_df

    def create_tree(self, roots, depth=None):

        if type(roots) is not list:
            roots = [roots]
        graph = pydot.Dot(graph_type='digraph', strict=True)

        def create_node(genre_id):
            title = self.df.at[genre_id, 'title']
            ntracks = self.df.at[genre_id, '#tracks']
            #name = self.df.at[genre_id, 'title'] + '\n' + str(genre_id)
            name = '"{}\n{} / {}"'.format(title, genre_id, ntracks)
            return pydot.Node(name)

        def create_tree(root_id, node_p, depth):
            if depth == 0:
                return
            children = self.df[self.df['parent'] == root_id]
            for child in children.iterrows():
                genre_id = child[0]
                node_c = create_node(genre_id)
                graph.add_edge(pydot.Edge(node_p, node_c))
                create_tree(genre_id, node_c,
                            depth-1 if depth is not None else None)

        for root in roots:
            node_p = create_node(root)
            graph.add_node(node_p)
            create_tree(root, node_p, depth)

        return graph

    def find_roots(self):
        roots = []
        for gid, row in self.df.iterrows():
            parent = row['parent']
            title = row['title']
            if parent == 0:
                roots.append(gid)
            elif parent not in self.df.index:
                msg = '{} ({}) has parent {} which is missing'.format(
                        gid, title, parent)
                raise RuntimeError(msg)
        return roots


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(int(track_id))
    path = os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')
    return path


class Loader:
    def load(self, filepath):
        raise NotImplemented()


class RawAudioLoader(Loader):
    def __init__(self, sampling_rate=SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.shape = (NB_AUDIO_SAMPLES * sampling_rate // SAMPLING_RATE, )

    def load(self, filepath):
        return self._load(filepath)[:self.shape[0]]


class LibrosaLoader(RawAudioLoader):
    def _load(self, filepath):
        import librosa
        sr = self.sampling_rate if self.sampling_rate != SAMPLING_RATE else None
        # kaiser_fast is 3x faster than kaiser_best
        #x, sr = librosa.load(filepath, sr=sr, res_type='kaiser_fast')
        x, sr = librosa.load(filepath, sr=sr)
        return x


class AudioreadLoader(RawAudioLoader):
    def _load(self, filepath):
        import audioread
        a = audioread.audio_open(filepath)
        a.read_data()


class PydubLoader(RawAudioLoader):
    def _load(self, filepath):
        from pydub import AudioSegment
        song = AudioSegment.from_file(filepath)
        song = song.set_channels(1)
        x = song.get_array_of_samples()
        return np.array(x)


class FfmpegLoader(RawAudioLoader):
    def _load(self, filepath):
        """Fastest and less CPU intensive loading method."""
        import subprocess as sp
        command = ['ffmpeg',
                   '-i', filepath,
                   '-f', 's16le',
                   '-acodec', 'pcm_s16le',
                   '-ac', '1']  # channels: 2 for stereo, 1 for mono
        if self.sampling_rate != SAMPLING_RATE:
            command.extend(['-ar', str(self.sampling_rate)])
        command.append('-')
        # 30s at 44.1 kHz ~= 1.3e6
        proc = sp.run(command, stdout=sp.PIPE, bufsize=10**7, stderr=sp.DEVNULL, check=True)

        return np.fromstring(proc.stdout, dtype="int16")


def build_sample_loader(audio_dir, Y, loader):

    class SampleLoader:

        def __init__(self, tids, batch_size=4):

            self.tids = tids
            np.random.shuffle(self.tids)
            self.batch_size = batch_size
            self.loader = loader
            X_shape = np.append(self.batch_size, loader.shape)
            self.X = np.empty(X_shape)
            self.Y = np.empty((self.batch_size, Y.shape[1]), dtype=np.bool)
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            tids = np.array(self.tids[self.idx:self.idx+self.batch_size])
            for tid in tids:
                try:
                    self.X[self.idx] = self.loader.load(get_audio_path(audio_dir, tid))
                    self.Y[self.idx] = Y.loc[tid]
                except:
                    continue
                self.idx += 1
                if self.idx % self.batch_size == 0: ## full
                    return self.X, self.Y
                self.idx %= (len(self.tids)-self.batch_size)
                if self.idx == 0: ## full epoch
                    np.random.shuffle(self.tids)

    return SampleLoader
