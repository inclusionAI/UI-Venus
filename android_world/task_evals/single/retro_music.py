# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been modified from the original AndroidWorld project.
# Changes: Implementing tasks in VenusBench.


"""Tasks for Retro Music app."""

import dataclasses
import os
import random
from typing import Any
from android_world.env import adb_utils
from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils


_APP_NAME = 'retro music'
_PLAYLIST_DB_PATH = (
    '/data/data/code.name.monkey.retromusic/databases/playlist.db'
)
_PLAYBACK_DB_PATH = (
    '/data/data/code.name.monkey.retromusic/databases/music_playback_state.db'
)


def _get_playlist_info_query() -> str:
  """Gets query for fetching playlists and their associated files."""
  # Note: Since we are using a window function, this requires sqlite 3.25.0.
  return """
    SELECT
        pe.playlist_name AS playlist_name,
        se.title AS media_file_name,
        se.duration AS duration_ms,
        ROW_NUMBER() OVER (
            PARTITION BY pe.playlist_name
            ORDER BY se.song_key
        ) - 1 AS order_in_playlist
    FROM
        PlaylistEntity pe
        JOIN SongEntity se ON pe.playlist_id = se.playlist_creator_id
    ORDER BY
        pe.playlist_name,
        order_in_playlist;
    """


def _get_playlist_data(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.PlaylistInfo]:
  """Executes join query to fetch playlist file info."""
  with env.controller.pull_file(
      _PLAYLIST_DB_PATH, timeout_sec=3
  ) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(_PLAYLIST_DB_PATH)[1]
    )
    return sqlite_utils.execute_query(
        _get_playlist_info_query(),
        local_db_path,
        sqlite_schema_utils.PlaylistInfo,
    )


def _get_playing_queue(env: interface.AsyncEnv) -> list[str]:
  """Executes join query to fetch playlist file info."""

  @dataclasses.dataclass(frozen=True)
  class Queue(sqlite_schema_utils.SQLiteRow):
    title: str

  with env.controller.pull_file(
      _PLAYBACK_DB_PATH, timeout_sec=3
  ) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(_PLAYBACK_DB_PATH)[1]
    )
    result = sqlite_utils.execute_query(
        'SELECT title from playing_queue;',
        local_db_path,
        Queue,
    )
    return [r.title for r in result]


def _clear_playlist_dbs(env: interface.AsyncEnv) -> None:
  """Clears all DBs related to playlists."""
  sqlite_utils.delete_all_rows_from_table(
      'PlaylistEntity', _PLAYLIST_DB_PATH, env, _APP_NAME
  )
  sqlite_utils.delete_all_rows_from_table(
      'SongEntity', _PLAYLIST_DB_PATH, env, _APP_NAME
  )


def _scan_music_directory(env: interface.AsyncEnv):
  """Scans the music directory to update the media store."""
  action = 'android.intent.action.MEDIA_SCANNER_SCAN_FILE'
  data_uri = 'file:///storage/emulated/0/Music'
  adb_utils.send_android_intent(
      command='broadcast', action=action, env=env.controller, data_uri=data_uri
  )
  adb_utils.close_app('retro music', env.controller)


class RetroCreatePlaylist(task_eval.TaskEval):
  """Task to create a playlist in Retro Music."""

  app_names = ['retro music']
  complexity = 2.4
  schema = {
      'type': 'object',
      'properties': {
          'playlist_name': {'type': 'string'},
          'files': {
              'type': 'array',
              'items': {'type': 'string'},
          },
      },
      'required': ['playlist_name', 'files'],
  }
  template = ''  # Directly use goal.

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'Create a playlist in Retro Music titled "{playlist_name}" with the'
        f' following songs, in order: {names}'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    _clear_playlist_dbs(env)

    for file in self.params['files'] + self.params['noise_files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=random.randint(3 * 60 * 1000, 5 * 60 * 1000),
      )
    _scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    actual = _get_playlist_data(env)
    return int(
        sqlite_validators.verify_playlist(
            actual,
            self.params['playlist_name'],
            [f.split('.')[0] for f in self.params['files']],
        )
    )

  def tear_down(self, env: interface.AsyncEnv):
    super().tear_down(env)
    user_data_generation.clear_internal_storage(env)
    _clear_playlist_dbs(env)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    files = [f'{name}.mp3' for name in random.sample(_SONGS, 15)]
    num_files = random.randint(2, 5)
    files, noise_files = files[0:num_files], files[num_files:]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': noise_files,
    }


class RetroPlayingQueue(RetroCreatePlaylist):
  """Task to create a playing queue in Retro Music."""

  complexity = 3.2

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    # files = [f'{name}.mp3' for name in random.sample(_SONGS, 15)]

    return (
        f'Add the following songs, in order, {names} to my playing queue in'
        ' Retro music.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    queue = _get_playing_queue(env)
    expected = [f.split('.')[0] for f in self.params['files']]
    return int(queue == expected)


class RetroSavePlaylist(RetroCreatePlaylist):
  """Task to create a playlist and save it in Retro Music."""

  complexity = 5

  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])
    playlist_name = self.params['playlist_name']
    return (
        f'Create a playlist in Retro Music titled "{playlist_name}" with the'
        f' following songs, in order: {names}. Then export the playlist to the'
        ' Downloads directory on the device.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    playlist_exists = file_utils.check_file_exists(
        file_utils.convert_to_posix_path(
            device_constants.DOWNLOAD_DATA,
            self.params['playlist_name'] + '.m3u',
        ),
        env.controller,
    )
    return (super().is_successful(env) + int(playlist_exists)) / 2.0



class MultiRoundRetroSavePlaylist(RetroCreatePlaylist):
  """Task to create a playlist and save it in Retro Music."""

  complexity = 10
  round = 0
  max_round = 5

  def initialize_task(self, env: interface.AsyncEnv):
    super().initialize_task(env)
    user_data_generation.clear_internal_storage(env)
    _clear_playlist_dbs(env)

    for file in self.params['files'] + self.params['noise_files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=random.randint(3 * 60 * 1000, 5 * 60 * 1000),
      )
    _scan_music_directory(env)

  @property
  def goal(self) -> str:
    if self.round == 0:
      names = ', '.join(f.split('.')[0] for f in self.params['files_1st'])
      playlist_name = self.params['playlist_name']
      return (
          f'Create a playlist in Retro Music titled "{playlist_name}" with the'
          f' following songs, in order: {names}. '
      )
    elif self.round == 1:
      return 'Then export the playlist to the Downloads directory on the device.'
    
    elif self.round == 2:
      names2 = ', '.join(f.split('.')[0] for f in self.params['files_2nd'])
      playlist_name2 = self.params['playlist_name2']
      # f'{playlist_name1}{playlist_name2}'
      return (
          f'Create a new playlist whose title is '
          f'the exact title of the previous playlist '
          f'concatenated with the string "{playlist_name2}".'
          f'This playlist contains the following songs, in order: {names2}. '
      )
    
    elif self.round == 3:
      return (
        'Delete the playlist that has not yet been exported to the Downloads folder.'
      )
    
    elif self.round == 4:
      return (
        'Restore the playlist you just deleted; do not delete it.'
      )
    

  def is_successful(self, env: interface.AsyncEnv) -> float:
    if self.round == 0:
      actual = _get_playlist_data(env)
      
      return int(
          sqlite_validators.verify_playlist(
              actual,
              self.params['playlist_name'],
              [f.split('.')[0] for f in self.params['files_1st']],
          )
      )
    elif self.round == 1:
      actual = _get_playlist_data(env)
      playlist_verified_in_retro = int(
          sqlite_validators.verify_playlist(
              actual,
              self.params['playlist_name'],
              [f.split('.')[0] for f in self.params['files_1st']],
          )
      )

      playlist_exists_in_download = file_utils.check_file_exists(
          file_utils.convert_to_posix_path(
              device_constants.DOWNLOAD_DATA,
              self.params['playlist_name'] + '.m3u',
          ),
          env.controller,
      )

      return playlist_verified_in_retro and playlist_exists_in_download
    elif self.round == 2:
      actual = _get_playlist_data(env)
      playlist_1st_verified_in_retro = int(
          sqlite_validators.verify_playlist(
              actual,
              self.params['playlist_name'],
              [f.split('.')[0] for f in self.params['files_1st']],
          )
      )

      playlist_2nd_verified_in_retro = int(
          sqlite_validators.verify_playlist(
              actual,
              f"{self.params['playlist_name']}{self.params['playlist_name2']}",
              [f.split('.')[0] for f in self.params['files_2nd']],
          )
      )
      return playlist_1st_verified_in_retro and playlist_2nd_verified_in_retro
    
    elif self.round == 3:
      actual = _get_playlist_data(env)

      playlist_1st_verified_in_retro = int(
          sqlite_validators.verify_playlist(
              actual,
              self.params['playlist_name'],
              [f.split('.')[0] for f in self.params['files_1st']],
          )
      )
      # 1st should not be deleted
      if not playlist_1st_verified_in_retro:
        return 0.0
      
      # 2nd should be deleted
      if sqlite_validators.playlist_exist(actual,self.params['playlist_name2']):
        return 0.0
      
      # return 1
    elif self.round == 4:
      actual = _get_playlist_data(env)
      playlist_1st_verified_in_retro = int(
          sqlite_validators.verify_playlist(
              actual,
              self.params['playlist_name'],
              [f.split('.')[0] for f in self.params['files_1st']],
          )
      )

      playlist_2nd_verified_in_retro = int(
          sqlite_validators.verify_playlist(
              actual,
              f"{self.params['playlist_name']}{self.params['playlist_name2']}",
              [f.split('.')[0] for f in self.params['files_2nd']],
          )
      )
      return playlist_1st_verified_in_retro and playlist_2nd_verified_in_retro
    
    return 1.0
  

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    playlist_name2 = _generate_playlist_name() + '_2nd'

    files = [f'{name}.mp3' for name in random.sample(_SONGS, 30)]
    num_files_1st = random.randint(2, 5)
    num_files_2nd = random.randint(2, 5)
    num_files_3rd = random.randint(2, 5)

    start_1st = 0
    end_1st = num_files_1st
    
    start_2nd = end_1st
    end_2nd = end_1st + num_files_2nd
    
    start_3rd = end_2nd
    end_3rd = end_2nd + num_files_3rd
    
    files_1st = files[start_1st:end_1st]
    files_2nd = files[start_2nd:end_2nd]
    files_3rd = files[start_3rd:end_3rd]
    noise_files = files[end_3rd:]

    return {
        'playlist_name': playlist_name,
        'playlist_name2': playlist_name2,
        'files': files,
        'files_1st': files_1st,
        'files_2nd': files_2nd,
        'files_3rd': files_3rd,
        'noise_files': noise_files,
    }


class MultiRoundRetroSavePlaylistCHS(MultiRoundRetroSavePlaylist):
  @property
  def goal(self) -> str:
    if self.round == 0:
      names = ', '.join(f.split('.')[0] for f in self.params['files'])
      playlist_name = self.params['playlist_name']
      return (
          f'在 Retro Music 中创建一个名为 "{playlist_name}" 的播放列表，按顺序包含以下歌曲：{names}。'
      )
    elif self.round == 1:
      return '然后将播放列表导出到设备的下载目录。'
    elif self.round == 2:
      names2 = ', '.join(f.split('.')[0] for f in self.params['files_2nd'])
      playlist_name2 = self.params['playlist_name2']
      return (
          f'创建一个新播放列表，其标题是上一个播放列表的确切标题加上字符串 "{playlist_name2}"。'
          f'该播放列表按顺序包含以下歌曲：{names2}。'
      )
    elif self.round == 3:
      return (
        '删除尚未导出到下载文件夹的播放列表。'
      )
    elif self.round == 4:
      return (
        '恢复刚刚删除的播放列表；不要删除它。'
      )

    
class MultiRoundRetroSavePlaylistVariation(MultiRoundRetroSavePlaylist):
  @property
  def goal(self) -> str:
    if self.round == 0:
      names = ', '.join(f.split('.')[0] for f in self.params['files'])
      playlist_name = self.params['playlist_name']
      return (
          f'In Retro Music, create a playlist titled "{playlist_name}" '
          'containing the following songs in order: {names}.'
      )
    elif self.round == 1:
      return 'Then export the playlist to the Downloads directory on the device.'
    elif self.round == 2:
      names2 = ', '.join(f.split('.')[0] for f in self.params['files_2nd'])
      playlist_name2 = self.params['playlist_name2']
      return (
          f'Create a new playlist whose title is the '
          f'exact title of the previous playlist concatenated with the string "{playlist_name2}".'
          f'This playlist contains the following songs, in order: {names2}.'
      )
    elif self.round == 3:
      return (
        'Remove the playlist that hasn\'t been exported to the Downloads folder yet.'
    )
    elif self.round == 4:
      return (
        'Recover the playlist you previously deleted; keep it intact.'
    )

def _generate_list_with_sum(n, m):
  """Generates a list of m integers with sum n."""
  random_numbers = [random.randint(0, n) for _ in range(m - 1)]
  random_numbers.sort()
  random_numbers.insert(0, 0)
  random_numbers.append(n)
  result = [random_numbers[i + 1] - random_numbers[i] for i in range(m)]
  return result


class RetroPlaylistDuration(RetroCreatePlaylist):
  """Task to create a playlist with a specific duration in Retro Music."""

  app_names = ['retro music']
  complexity = 3

  @property
  def goal(self) -> str:
    return (
        'Create a playlist in Retro Music titled'
        f' "{self.params["playlist_name"]}" with a duration between 45 and 50'
        ' minutes using the provided songs.'
    )

  def initialize_task(self, env: interface.AsyncEnv):
    _clear_playlist_dbs(env)

    # Guarantee there is an answer.
    durations = _generate_list_with_sum(
        int(47.5 * 60 * 1000), len(self.params['files'])
    )
    for file, duration in zip(self.params['files'], durations):
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=duration,
      )

    for file in self.params['noise_files']:
      user_data_generation.write_mp3_file_to_device(
          file_utils.convert_to_posix_path(device_constants.MUSIC_DATA, file),
          env,
          title=file.split('.')[0],
          artist=random.choice(user_data_generation.COMMON_GIVEN_NAMES),
          duration_milliseconds=random.randint(3 * 60 * 1000, 5 * 60 * 1000),
      )
    _scan_music_directory(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    songs = _get_playlist_data(env)
    total_ms = 0
    for song in songs:
      if song.playlist_name != self.params['playlist_name']:
        return False
      total_ms += song.duration_ms
    return float(45 * 60 * 1000 <= total_ms <= 50 * 60 * 1000)

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    playlist_name = _generate_playlist_name()
    files = [f'{name}.mp3' for name in random.sample(_SONGS, 15)]
    num_files = random.randint(9, 11)
    files, noise_files = files[0:num_files], files[num_files:]
    return {
        'playlist_name': playlist_name,
        'files': files,
        'noise_files': noise_files,
    }


_SONGS = [
    'My Heart is Yours',
    'Endless Summer',
    'Whispering Wind',
    'Lost in the Echo',
    'Chasing Shadows',
    'Night Drive',
    'Echoes of Silence',
    'Bright Lights',
    'Moments',
    'Forever Young',
    'Rising Sun',
    'Silent Dreams',
    'City of Stars',
    'Moonlight Sonata',
    'Through the Storm',
    'Return to Paradise',
    'Voices in the Hall',
    'Under the Sky',
    "Dreamer's Awake",
    'Serenity Now',
    'Falling Feathers',
    'Orbiting Stars',
    'Reflections',
    'Beyond the Horizon',
    'Golden Days',
    'Twilight Calling',
    'Heartbeat Away',
    'Mystic Journey',
    'Hidden Paths',
    'Distant Memories',
    'Path to Zenith',
    'Eternal Flame',
    'Shadows of Time',
    'Whispers of the Past',
    'Waves of Change',
]


def _generate_playlist_name() -> str:
  """Generates a diverse and creative playlist name."""
  themes = [
      'Chill Beats',
      'Morning Vibes',
      'Workout Energy',
      'Study Sessions',
      'Golden Oldies',
      'Indie Gems',
      'Rock Anthems',
      'Electronic Waves',
      'Jazz Classics',
      'Hip Hop Hits',
      'Country Roads',
      'Classical Moods',
      'Pop Essentials',
      'Latin Grooves',
      'Reggae Rhythms',
      'Soulful Sounds',
      'Blues Vibes',
      'Metal Mayhem',
      'Party Mix',
      'Tranquil Tunes',
      'R&B Favorites',
      'Folk Inspirations',
      'Disco Nights',
      'Global Beats',
      'Sleepytime Songs',
      'Acoustic Sessions',
      'Vintage Vinyl',
      'Instrumental Study',
      'Coffeehouse Jazz',
      'Rainy Day Relax',
      'Gym Pump Up',
      'Retro Pop Hits',
      'Indie Rock Roadtrip',
      'Electronic Chillout',
      'Classical Concentration',
      'Jazz Lounge',
      'Hip Hop Bangers',
      'Country Classics',
      'Classical Opera Highlights',
      'Pop Punk Power',
      'Latin Dance Party',
      'Reggae Sunsplash',
      'Soul Classics',
      'Blues Break',
      'Party Starters',
      'Tranquil Ambient',
      'R&B Grooves',
      'Folk Favourites',
      'Disco Fever',
      'World Music Tour',
  ]
  identifier = random.randint(1, 999)

  theme = random.choice(themes)
  return f'{theme} {identifier}'

from android_world.policy.verification import VerifyPolicy
from typing import List

class RefusalRetroPlayingQueueConflict1(RetroCreatePlaylist):
    """Task to create a playing queue in Retro Music."""

    complexity = 3

    @property
    def goal(self) -> str:
        # 已经给出的歌名
        given_names = [f.split('.')[0] for f in self.params['files']]
        extra_names = self.params['extra_names']

        # 最终顺序：先给定的，再随机追加的
        names = ', '.join(given_names + extra_names)

        return (
            f'Add the following songs, in order, {names} to my playing queue in'
            ' Retro music.'
        )

    def is_successful(self, env: interface.AsyncEnv) -> float:
        if not env.interaction_cache:
          return 0.0
        extra_names = self.params['extra_names']

        def _format_names(names: List[str]) -> str:
          if len(names) == 1:
              return f'"{names[0]}"'
          return ', '.join(f'"{n}"' for n in names[:-1]) + f' and "{names[-1]}"'

        answer = (
          f"I can't add those songs to the queue because "
          f"{_format_names(extra_names)} {'is' if len(extra_names) == 1 else 'are'} "
          f"not found in Retro Music."
        )

        ply = VerifyPolicy({})
        result = ply.verify_textual_prediction(
            goal=self.goal,
            prediction=env.interaction_cache,
            gt_answer= answer,
            judgement_type = 'refusal'
        )
        return result
        
    

    @classmethod
    def generate_random_params(cls) -> dict[str, Any]:
      playlist_name = _generate_playlist_name()
      files = [f'{name}.mp3' for name in random.sample(_SONGS, 15)]
      num_files = random.randint(2, 5)
      files, noise_files = files[0:num_files], files[num_files:]


      # 已经给出的歌名
      given_names = [f.split('.')[0] for f in files]

      # 候选新歌名：不在 given_names 里的所有歌
      candidates = [s for s in _SONGS if s not in given_names]

      # 随机再取 1~3 首新歌
      k = random.randint(1, 3)
      extra_names = random.sample(candidates, k)



      return {
          'playlist_name': playlist_name,
          'files': files,
          'noise_files': noise_files,
          'extra_names': extra_names,
      }


class NoiseResistRetroPlayingQueueAPPCollapse(RetroCreatePlaylist):
  """Task to create a playing queue in Retro Music."""

  complexity = 4.2

  # Kill app condition and program
  collapse_steps = [5, 12]
  def check_status(self, env: interface.AsyncEnv):
    self.is_collapse_condition(env)

  def is_collapse_condition(
      self,
      env: interface.AsyncEnv,
  ) -> float:
    if self.current_step in self.collapse_steps:
      adb_utils.close_app(_APP_NAME, env.controller)

# original verification program
  @property
  def goal(self) -> str:
    names = ', '.join(f.split('.')[0] for f in self.params['files'])

    return (
        f'Add the following songs, in order, {names} to my playing queue in'
        ' Retro music.'
    )

  def is_successful(self, env: interface.AsyncEnv) -> float:
    queue = _get_playing_queue(env)
    expected = [f.split('.')[0] for f in self.params['files']]
    return int(queue == expected)