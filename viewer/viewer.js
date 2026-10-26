const DAY = 24000;
const BLOCK = {
  grass_block: '#7aa35a', gravel: '#a39d95', water: '#3d74c2', dirt_path: '#c09a5e',
  cobblestone_wall: '#7e7e7c', oak_slab: '#a07a4c', oak_leaves: '#3c6a2e',
  oak_planks: '#c9a46c', stone: '#8d8d8b', white_wool: '#ecebe6',
  stone_bricks: '#6c6d6b', oak_log: '#6a4b28', oak_sign: '#d8b77d',
  oak_wall_sign: '#d8b77d', chest: '#a8742c', bell: '#e8b830', lantern: '#ffd85a', air: '#2a2d2c',
};
const AGENT = {
  mayor: '#7b3fa0', builder: '#d0602a', freight: '#167a8c',
  security: '#2a4fb0', envoy: '#b8233a', investigator: '#2f8a3e',
};
const byId = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[character]);
const describe = value => value == null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const humanize = value => String(value ?? 'unknown').replace(/^minecraft:/, '').replace(/_/g, ' ');
const shortBlock = value => String(value ?? 'air').replace(/^minecraft:/, '').split('[')[0];
const activityColor = activity => String(activity).startsWith('travel') ? '#f08c1c'
  : activity === 'work' ? '#2563eb' : activity === 'school' ? '#9333ea'
    : activity === 'market' ? '#16a34a' : '#6b7280';
const agentColor = name => AGENT[name] || '#6b7280';
const clock = time => {
  const day = Math.floor(time / DAY);
  const minutes = Math.floor((time % DAY) * 1440 / DAY + 360) % 1440;
  return `day ${day}, ${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}`;
};
const number = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const position = value => {
  if (!Array.isArray(value) || value.length < 2) return null;
  const coordinates = [Number(value[0]), Number(value[value.length >= 3 ? 2 : 1])];
  return coordinates.every(Number.isFinite) ? coordinates : null;
};
const pill = value => value === true ? '<span class="pill good">Yes</span>'
  : value === false ? '<span class="pill bad">No</span>' : '<span class="pill pending">Pending / unknown</span>';

function upperBound(rows, time, getTime = row => row[0]) {
  let low = 0;
  let high = rows.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (getTime(rows[middle]) <= time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function lowerBound(rows, time, getTime = row => row[0]) {
  let low = 0;
  let high = rows.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (getTime(rows[middle]) < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** Same-origin JSON transport with encoded run names, cancellation and timeouts. */
class Api {
  /** Fetch JSON without a browser cache; errors retain their HTTP status.
   * @param {string} path Same-origin API path.
   * @param {object} [parameters] Query parameters.
   * @param {AbortSignal} [signal] Owning load's cancellation signal.
   * @returns {Promise<any>} Parsed response.
   */
  async get(path, parameters = {}, signal) {
    const url = new URL(path, window.location.href);
    for (const [key, value] of Object.entries(parameters)) {
      if (value !== '' && value != null) url.searchParams.set(key, String(value));
    }
    const controller = new AbortController();
    let timedOut = false;
    const cancel = () => controller.abort();
    if (signal?.aborted) cancel();
    signal?.addEventListener('abort', cancel, { once: true });
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 60000);
    try {
      const response = await fetch(url, { signal: controller.signal, cache: 'no-store', headers: { Accept: 'application/json' } });
      let validJson = true;
      const body = await response.json().catch(() => { validJson = false; return null; });
      if (!response.ok) {
        const error = new Error(body?.error || body?.message || `Request failed (${response.status}): ${path}`);
        error.status = response.status;
        throw error;
      }
      if (!validJson) throw new Error(`Expected JSON from ${path}. Is the viewer server running?`);
      return body;
    } catch (error) {
      if (timedOut) throw new Error('The local server took too long to respond. Retrying automatically.');
      throw error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', cancel);
    }
  }

  /** Fetch a run endpoint, propagating the selected scenario override.
   * @param {string} name Raw run directory name.
   * @param {string} endpoint Endpoint suffix.
   * @param {string} variant Empty for auto-detection.
   * @param {object} [parameters] Additional query parameters.
   * @param {AbortSignal} [signal] Load cancellation signal.
   * @returns {Promise<any>} Parsed response.
   */
  run(name, endpoint, variant, parameters = {}, signal) {
    return this.get(`/api/runs/${encodeURIComponent(name)}/${endpoint}`, { ...parameters, variant }, signal);
  }
}

/** Atomic run snapshots and step-held, binary-searched historical observations. */
class RunStore extends EventTarget {
  #controller = null;
  #busy = false;

  /** @param {Api} api Local API transport. */
  constructor(api) {
    super();
    this.api = api;
    this.name = '';
    this.variant = '';
    this.data = null;
    this.time = 0;
    this.follow = false;
    this.selectedAgent = '';
    this.loading = false;
    this.error = '';
    this.generation = 0;
    this.revision = 0;
    this.updatedAt = null;
  }

  /** Select a run/variant and invalidate all pending loads from the old selection.
   * @param {string} name Raw run name.
   * @param {string} [variant] Scenario override or empty string.
   * @returns {Promise<void>} Resolves after the first load attempt.
   */
  async load(name, variant = '') {
    this.#controller?.abort();
    this.#controller = new AbortController();
    this.generation += 1;
    this.name = name;
    this.variant = variant;
    this.data = null;
    this.selectedAgent = '';
    this.time = 0;
    this.error = '';
    this.loading = true;
    this.#busy = false;
    this.dispatchEvent(new Event('reset'));
    await this.refresh(true);
  }

  /** Refresh live/following runs; preserve the last good snapshot on failure.
   * @param {boolean} [force] Also refresh a recorded, non-following run.
   * @returns {Promise<void>} Resolves after committing or rejecting this refresh.
   */
  async refresh(force = false) {
    if (!this.name || this.#busy || (!force && !this.follow && !this.data?.meta.live && !this.error)) return;
    this.#busy = true;
    const generation = this.generation;
    const signal = this.#controller.signal;
    const request = (endpoint, parameters) => this.api.run(this.name, endpoint, this.variant, parameters, signal);
    this.dispatchEvent(new Event('status'));
    try {
      const meta = await request('meta');
      if (generation !== this.generation) return;
      meta.start = number(meta.start);
      meta.end = Math.max(meta.start, number(meta.end, meta.start));
      meta.agents = Array.isArray(meta.agents) ? meta.agents : [];
      meta.villagers = Array.isArray(meta.villagers) ? meta.villagers : [];
      meta.places = Array.isArray(meta.places) ? meta.places : [];
      const previous = this.data;
      let changed = force || !previous || previous.meta.modified !== meta.modified
        || previous.meta.start !== meta.start || previous.meta.end !== meta.end;
      const readMap = () => request('map').then(map => ({ map, mapIssue: '' })).catch(error => {
        if (signal.aborted) throw error;
        return { map: null, mapIssue: error.status === 404
          ? `No base map found. Add runs/village_${meta.variant || this.variant || 'sabotage'}/viewer_map.json; people and the other panels still work.`
          : `Terrain unavailable: ${error.message}. People and the other panels still work; the map will retry.` };
      });
      let snapshot = previous;
      if (changed) {
        const mapRequest = previous?.map ? Promise.resolve({ map: previous.map, mapIssue: '' })
          : readMap();
        const [mapResult, changes, tracks, events] = await Promise.all([
          mapRequest, request('changes'), request('tracks'),
          request('events', { from: meta.start, to: meta.end }),
        ]);
        if (generation !== this.generation) return;
        const sortedEvents = (Array.isArray(events) ? events : []).filter(event => Number.isFinite(event.t))
          .sort((first, second) => first.t - second.t);
        const decisions = new Map();
        const speech = [];
        for (const event of sortedEvents) {
          if (event.type === 'decision') {
            const name = String(event.agent ?? event.bot ?? event.actor ?? 'unknown');
            if (!decisions.has(name)) decisions.set(name, []);
            decisions.get(name).push(event);
          } else if (event.type === 'utterance') speech.push(event);
        }
        for (const group of ['agents', 'npcs', 'trips']) {
          tracks[group] ||= {};
          for (const rows of Object.values(tracks[group])) rows.sort((first, second) => first[0] - second[0]);
        }
        snapshot = { ...mapResult, changes: Array.isArray(changes) ? changes : [], tracks,
          events: sortedEvents, decisions, speech,
          agentChanges: (meta.agent_changes || []).slice().sort((first, second) => first[0] - second[0]) };
        snapshot.changes.sort((first, second) => first[0] - second[0]);
        this.revision += 1;
      } else if (!previous.map) {
        const mapResult = await readMap();
        if (generation !== this.generation) return;
        changed = mapResult.map !== previous.map || mapResult.mapIssue !== previous.mapIssue;
        snapshot = { ...previous, ...mapResult };
        if (mapResult.map) {
          const changes = await request('changes');
          if (generation !== this.generation) return;
          snapshot.changes = changes.slice().sort((first, second) => first[0] - second[0]);
          snapshot.agentChanges = (meta.agent_changes || []).slice().sort((first, second) => first[0] - second[0]);
          this.revision += 1;
        }
      }
      this.data = { ...snapshot, meta };
      this.time = !previous || this.follow ? meta.end : Math.max(meta.start, Math.min(meta.end, this.time));
      if (!meta.agents.some(agent => agent.name === this.selectedAgent)) {
        this.selectedAgent = meta.agents.find(agent => agent.name === 'builder')?.name || meta.agents[0]?.name || '';
      }
      this.error = '';
      this.updatedAt = new Date();
      this.dispatchEvent(new CustomEvent('data', { detail: { changed } }));
      this.dispatchEvent(new CustomEvent('time', { detail: { manual: false } }));
    } catch (error) {
      if (generation === this.generation && !signal.aborted) {
        this.error = `${error.message} Keeping any loaded data; retrying in 5 seconds.`;
      }
    } finally {
      if (generation === this.generation) {
        this.#busy = false;
        this.loading = false;
        this.dispatchEvent(new Event('status'));
      }
    }
  }

  /** Seek to an integer tick, optionally exiting follow mode.
   * @param {number} time Target tick.
   * @param {boolean} [manual] User-initiated seeking pauses playback.
   */
  setTime(time, manual = false) {
    if (!this.data) return;
    if (manual) this.setFollow(false);
    const { start, end } = this.data.meta;
    this.time = Math.max(start, Math.min(end, Math.floor(number(time, start))));
    this.dispatchEvent(new CustomEvent('time', { detail: { manual } }));
  }

  /** Toggle live following and snap to the latest known tick when enabled.
   * @param {boolean} enabled Whether to follow the growing run.
   */
  setFollow(enabled) {
    this.follow = enabled;
    this.dispatchEvent(new Event('follow'));
    if (enabled && this.data) this.setTime(this.data.meta.end);
  }

  /** Select an agent consistently across map and panels.
   * @param {string} name Agent role name.
   */
  selectAgent(name) {
    if (!this.data?.meta.agents.some(agent => agent.name === name)) return;
    this.selectedAgent = name;
    this.dispatchEvent(new Event('selection'));
  }

  /** Read the latest observed agent position, never a future track sample.
   * @param {string} name Agent role name.
   * @param {number} [time] Tick to inspect.
   * @returns {object} Position, sample time, skill and state.
   */
  agentAt(name, time = this.time) {
    const rows = this.data?.tracks.agents[name] || [];
    const index = upperBound(rows, time) - 1;
    const sample = rows[index];
    const agent = this.data?.meta.agents.find(candidate => candidate.name === name);
    return { pos: sample ? [sample[1], sample[2]] : position(agent?.spawn),
      observed: sample?.[0], skill: sample?.[3] || 'unknown', state: sample?.[4] || 'No observation yet', index };
  }

  /** Infer villager activity from a known trip, separately from observed position.
   * Null arrivals are terminal/unreachable, not perpetually active journeys.
   * @param {string} id Raw villager identifier.
   * @param {number} [time] Tick to inspect.
   * @returns {object} Step-held position, inferred activity and time-safe trip status.
   */
  villagerAt(id, time = this.time) {
    const rows = this.data?.tracks.npcs[id] || [];
    const index = upperBound(rows, time) - 1;
    const sample = rows[index];
    const trips = this.data?.tracks.trips[id] || [];
    const tripIndex = upperBound(trips, time) - 1;
    const trip = trips[tripIndex];
    const villager = this.data?.meta.villagers.find(candidate => String(candidate.id) === String(id));
    const arrived = Boolean(trip && trip[1] != null && Number.isFinite(Number(trip[1])) && time >= Number(trip[1]));
    const travelling = Boolean(trip && trip[1] != null && Number.isFinite(Number(trip[1])) && !arrived);
    const destination = trip?.[3] ?? villager?.home;
    const activity = !trip ? 'home' : travelling ? 'travelling' : arrived ? String(trip[2] || 'unknown') : 'unknown';
    const fallbackPlace = arrived ? destination : !trip ? villager?.home : trip[6] || villager?.home;
    const place = this.data?.meta.places.find(candidate => candidate.name === fallbackPlace);
    return { pos: sample ? [sample[1], sample[2]] : position(place?.pos), observed: sample?.[0],
      index, tripIndex, trip, destination, activity, travelling, arrived,
      onTime: arrived ? trip[4] : null, outcome: arrived ? trip[5] : null,
      status: !trip ? 'Home inferred; no trip known' : travelling ? 'In transit; arrival pending'
        : arrived ? 'Last completed trip; activity inferred' : 'Failed trip; no recorded arrival, activity unknown' };
  }

  /** Return up to ten decisions at or before the requested tick, including waits.
   * @param {string} name Agent role name.
   * @param {number} [time] Tick to inspect.
   * @returns {object[]} Chronological decision history.
   */
  decisionsAt(name, time = this.time) {
    const rows = this.data?.decisions.get(name) || [];
    const end = upperBound(rows, time, event => event.t);
    return rows.slice(Math.max(0, end - 10), end);
  }
}

/** Game-clock transport, accessible seeking and frame-rate-independent playback. */
class Timeline {
  #frame = null;
  #lastFrame = null;
  #cursor = 0;

  /** @param {RunStore} store Shared historical run state. */
  constructor(store) {
    this.store = store;
    this.playing = false;
    byId('time-slider').addEventListener('input', event => store.setTime(Number(event.target.value), true));
    byId('play').addEventListener('click', () => this.toggle());
    byId('step-back').addEventListener('click', () => this.step(-40));
    byId('step-forward').addEventListener('click', () => this.step(40));
    byId('jump-start').addEventListener('click', () => store.setTime(store.data.meta.start, true));
    byId('jump-end').addEventListener('click', () => store.setTime(store.data.meta.end, true));
    byId('follow-live').addEventListener('change', event => {
      this.pause();
      store.setFollow(event.target.checked);
      if (event.target.checked) void store.refresh(true);
    });
    store.addEventListener('time', event => {
      if (event.detail.manual) this.pause();
      this.render();
    });
    store.addEventListener('reset', () => { this.pause(); this.render(); });
    store.addEventListener('follow', () => {
      if (store.follow) this.pause();
      byId('follow-live').checked = store.follow;
    });
    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey
        || event.target.closest('input, select, textarea, button, summary, a, [contenteditable="true"]')) return;
      if (event.code === 'Space') {
        event.preventDefault();
        if (!event.repeat) this.toggle();
      } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        this.step((event.key === 'ArrowLeft' ? -1 : 1) * (event.shiftKey ? 1200 : 40));
      }
    });
    this.render();
  }

  /** Start or pause playback; restarting at the end replays from the beginning. */
  toggle() {
    if (this.playing) { this.pause(); return; }
    if (!this.store.data || this.store.data.meta.start === this.store.data.meta.end) return;
    this.store.setFollow(false);
    if (this.store.time >= this.store.data.meta.end) this.store.setTime(this.store.data.meta.start);
    this.#cursor = this.store.time;
    this.#lastFrame = null;
    this.playing = true;
    this.#frame = requestAnimationFrame(time => this.#advance(time));
    this.render();
  }

  /** Pause playback without changing the selected tick. */
  pause() {
    if (this.#frame != null) cancelAnimationFrame(this.#frame);
    this.#frame = null;
    this.#lastFrame = null;
    this.playing = false;
    this.render();
  }

  /** Move by an exact number of ticks.
   * @param {number} ticks Signed tick distance.
   */
  step(ticks) { this.store.setTime(this.store.time + ticks, true); }

  /** Synchronize clock, slider, range and playback controls with the store. */
  render() {
    const { data, time } = this.store;
    const slider = byId('time-slider');
    const playable = Boolean(data && data.meta.end > data.meta.start);
    for (const id of ['time-slider', 'play', 'step-back', 'step-forward', 'jump-start', 'jump-end']) byId(id).disabled = !playable;
    slider.min = data?.meta.start ?? 0;
    slider.max = data?.meta.end ?? 1;
    slider.value = time;
    slider.setAttribute('aria-valuetext', `${clock(time)}, tick ${time}`);
    byId('clock').textContent = clock(time);
    byId('tick').textContent = `tick ${time.toLocaleString()}`;
    byId('run-range').textContent = data ? `${clock(data.meta.start)} → ${clock(data.meta.end)}` : 'Choose a run to begin';
    byId('play').textContent = this.playing ? 'Pause' : 'Play';
    byId('play').setAttribute('aria-label', this.playing ? 'Pause replay' : 'Play replay');
    byId('play').setAttribute('aria-pressed', String(this.playing));
  }

  #advance(timestamp) {
    if (!this.playing || !this.store.data) return;
    if (this.#lastFrame != null) this.#cursor += Math.min(250, timestamp - this.#lastFrame) / 1000 * 20 * Number(byId('speed').value);
    this.#lastFrame = timestamp;
    this.store.setTime(this.#cursor);
    if (this.store.time >= this.store.data.meta.end) this.pause();
    else this.#frame = requestAnimationFrame(time => this.#advance(time));
  }
}

/** Pixelated layered terrain, reversible reconstruction, historical overlays and hit-testing. */
class MapView {
  #canvas = byId('map-canvas');
  #context = this.#canvas.getContext('2d');
  #terrain = document.createElement('canvas');
  #terrainContext = this.#terrain.getContext('2d');
  #base = null;
  #layers = [];
  #layerByY = new Map();
  #palette = [];
  #paletteIndex = new Map();
  #image = null;
  #top = [];
  #topY = [];
  #applied = 0;
  #dirtyAll = true;
  #frame = null;
  #width = 1;
  #height = 1;
  #pointer = null;
  #hover = null;
  #people = [];
  #selectionKey = '';
  #origin = -128;
  #size = 256;
  #zoom = 1;
  #center = [0, 0];
  #colors = {};

  /** @param {RunStore} store Shared historical run state. */
  constructor(store) {
    this.store = store;
    this.overlays = { npcs: true, agents: true, speech: true, changes: true, places: true, routes: false };
    store.addEventListener('data', event => {
      if (event.detail.changed) this.setData();
      else this.render();
    });
    store.addEventListener('reset', () => this.setData());
    store.addEventListener('time', () => this.render());
    store.addEventListener('selection', () => this.render());
    document.querySelectorAll('[data-overlay]').forEach(input => input.addEventListener('change', () => {
      this.overlays[input.dataset.overlay] = input.checked;
      this.render();
    }));
    byId('zoom-in').addEventListener('click', () => this.zoom(1.4));
    byId('zoom-out').addEventListener('click', () => this.zoom(1 / 1.4));
    byId('map-fit').addEventListener('click', () => this.fit());
    this.#canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const point = this.#localPoint(event);
      this.zoom(Math.exp(-Math.max(-150, Math.min(150, event.deltaY)) * .007), point);
    }, { passive: false });
    this.#canvas.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      this.#canvas.focus({ preventScroll: true });
      this.#canvas.setPointerCapture(event.pointerId);
      this.#pointer = { id: event.pointerId, x: event.clientX, y: event.clientY,
        startX: event.clientX, startY: event.clientY, moved: false };
      this.#canvas.classList.add('dragging');
      byId('map-tooltip').hidden = true;
    });
    this.#canvas.addEventListener('pointermove', event => {
      this.#hover = this.#localPoint(event);
      if (this.#pointer?.id === event.pointerId) {
        const scale = this.#scale();
        this.#center[0] -= (event.clientX - this.#pointer.x) / scale;
        this.#center[1] -= (event.clientY - this.#pointer.y) / scale;
        this.#pointer.x = event.clientX;
        this.#pointer.y = event.clientY;
        this.#pointer.moved ||= Math.hypot(event.clientX - this.#pointer.startX, event.clientY - this.#pointer.startY) > 4;
        this.render();
      } else this.#tooltip();
    });
    this.#canvas.addEventListener('pointerup', event => {
      if (this.#pointer?.id !== event.pointerId) return;
      const moved = this.#pointer.moved;
      this.#pointer = null;
      this.#canvas.classList.remove('dragging');
      if (this.#canvas.hasPointerCapture(event.pointerId)) this.#canvas.releasePointerCapture(event.pointerId);
      if (!moved) {
        const hit = this.hitTest(...this.#localPoint(event));
        if (hit.person?.kind === 'agent') store.selectAgent(hit.person.name);
      }
      this.#tooltip();
    });
    this.#canvas.addEventListener('pointercancel', () => {
      this.#pointer = null;
      this.#hover = null;
      this.#canvas.classList.remove('dragging');
      byId('map-tooltip').hidden = true;
    });
    this.#canvas.addEventListener('pointerleave', () => {
      this.#hover = null;
      byId('map-tooltip').hidden = true;
    });
    this.#canvas.addEventListener('keydown', event => {
      const key = event.key.toLowerCase();
      if (key === '+' || key === '=') this.zoom(1.4);
      else if (key === '-') this.zoom(1 / 1.4);
      else if (key === 'home') this.fit();
      else if (['w', 'a', 's', 'd'].includes(key)) {
        this.#center[0] += (key === 'a' ? -40 : key === 'd' ? 40 : 0) / this.#scale();
        this.#center[1] += (key === 'w' ? -40 : key === 's' ? 40 : 0) / this.#scale();
        this.render();
      } else return;
      event.preventDefault();
    });
    new ResizeObserver(() => this.#resize()).observe(this.#canvas);
    const theme = window.matchMedia('(prefers-color-scheme: dark)');
    theme.addEventListener('change', () => { this.#readColors(); this.render(); });
    this.#readColors();
    this.#resize();
    this.#legend();
  }

  /** Adopt a fresh run snapshot and reset terrain, preserving the same run's camera. */
  setData() {
    const map = this.store.data?.map;
    this.#base = map && Number.isFinite(map.origin) && Number.isInteger(map.size) && map.size > 0
      && Array.isArray(map.palette) && Array.isArray(map.layers) && map.layers.length ? map : null;
    this.#origin = this.#base?.origin ?? -128;
    this.#size = this.#base?.size ?? 256;
    this.#palette = this.#base?.palette.slice() || [];
    this.#paletteIndex = new Map(this.#palette.map((block, index) => [block, index]));
    this.#terrain.width = this.#size;
    this.#terrain.height = this.#size;
    this.#image = this.#terrainContext.createImageData(this.#size, this.#size);
    this.#top = Array(this.#size * this.#size).fill('air');
    this.#topY = Array(this.#size * this.#size).fill(null);
    this.#resetLayers();
    const key = `${this.store.name}\0${this.store.variant}`;
    if (key !== this.#selectionKey) {
      this.#selectionKey = key;
      this.fit();
    }
    const notice = byId('map-empty');
    notice.hidden = Boolean(this.#base);
    notice.textContent = !this.store.data ? this.store.name ? 'Loading map and historical observations…' : 'Choose a run to explore its map.'
      : this.store.data.mapIssue || 'The base map has no usable layers. People and other panels remain available.';
    this.#legend();
    this.render();
  }

  /** Recenter and fit the entire map to the viewport. */
  fit() {
    this.#zoom = 1;
    this.#center = [this.#origin + this.#size / 2, this.#origin + this.#size / 2];
    this.render();
  }

  /** Zoom around a screen-space anchor, keeping its world position fixed.
   * @param {number} factor Multiplicative zoom change.
   * @param {number[]} [anchor] CSS-pixel position within the canvas.
   */
  zoom(factor, anchor = [this.#width / 2, this.#height / 2]) {
    const before = this.#world(...anchor);
    this.#zoom = Math.max(.5, Math.min(32, this.#zoom * factor));
    const after = this.#world(...anchor);
    this.#center[0] += before[0] - after[0];
    this.#center[1] += before[1] - after[1];
    this.render();
  }

  /** Resolve terrain coordinates and the closest visible person at a screen point.
   * @param {number} screenX Horizontal CSS-pixel coordinate.
   * @param {number} screenY Vertical CSS-pixel coordinate.
   * @returns {object} World cell, top block and nearest visible person.
   */
  hitTest(screenX, screenY) {
    const [worldX, worldZ] = this.#world(screenX, screenY);
    const cellX = Math.floor(worldX);
    const cellZ = Math.floor(worldZ);
    const column = cellX - this.#origin;
    const row = cellZ - this.#origin;
    const index = row * this.#size + column;
    const inside = column >= 0 && row >= 0 && column < this.#size && row < this.#size;
    let person = null;
    let closest = Math.max(4, 13 / this.#scale());
    for (const candidate of this.#people) {
      const distance = Math.hypot(candidate.pos[0] - worldX, candidate.pos[1] - worldZ);
      if (distance < closest) { person = candidate; closest = distance; }
    }
    return { x: cellX, z: cellZ, y: inside ? this.#topY[index] : null,
      block: this.#base && inside ? this.#top[index] : null, person };
  }

  /** Schedule one coalesced terrain and overlay render. */
  render() {
    if (this.#frame != null) return;
    this.#frame = requestAnimationFrame(() => {
      this.#frame = null;
      this.#draw();
    });
  }

  #readColors() {
    const styles = getComputedStyle(document.documentElement);
    this.#colors = Object.fromEntries(['bg', 'panel', 'ink', 'muted', 'rule'].map(key => [key, styles.getPropertyValue(`--${key}`).trim()]));
  }

  #resize() {
    const rectangle = this.#canvas.getBoundingClientRect();
    this.#width = Math.max(1, rectangle.width);
    this.#height = Math.max(1, rectangle.height);
    const ratio = window.devicePixelRatio || 1;
    this.#canvas.width = Math.round(this.#width * ratio);
    this.#canvas.height = Math.round(this.#height * ratio);
    this.#context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.render();
  }

  #scale() { return Math.min(this.#width, this.#height) / this.#size * this.#zoom; }

  #project(worldX, worldZ) {
    const scale = this.#scale();
    return [(worldX - this.#center[0]) * scale + this.#width / 2, (worldZ - this.#center[1]) * scale + this.#height / 2];
  }

  #world(screenX, screenY) {
    const scale = this.#scale();
    return [(screenX - this.#width / 2) / scale + this.#center[0], (screenY - this.#height / 2) / scale + this.#center[1]];
  }

  #localPoint(event) {
    const rectangle = this.#canvas.getBoundingClientRect();
    return [event.clientX - rectangle.left, event.clientY - rectangle.top];
  }

  #resetLayers() {
    this.#layers = (this.#base?.layers || []).slice().sort((first, second) => second.y - first.y)
      .map(layer => ({ y: layer.y, grid: layer.grid.slice() }));
    this.#layerByY = new Map(this.#layers.map(layer => [layer.y, layer]));
    this.#applied = 0;
    this.#dirtyAll = true;
  }

  #syncTerrain() {
    if (!this.#base || !this.store.data) return;
    const { changes } = this.store.data;
    const count = upperBound(changes, this.store.time);
    if (count < this.#applied) this.#resetLayers();
    const columns = new Set();
    for (; this.#applied < count; this.#applied += 1) {
      const [, worldX, worldY, worldZ, block] = changes[this.#applied];
      const layer = this.#layerByY.get(worldY);
      const column = worldX - this.#origin;
      const row = worldZ - this.#origin;
      if (!layer || column < 0 || row < 0 || column >= this.#size || row >= this.#size) continue;
      if (!this.#paletteIndex.has(block)) {
        this.#paletteIndex.set(block, this.#palette.length);
        this.#palette.push(block);
      }
      const index = row * this.#size + column;
      layer.grid[index] = this.#paletteIndex.get(block);
      columns.add(index);
    }
    if (this.#dirtyAll) {
      for (let index = 0; index < this.#size * this.#size; index += 1) this.#paintColumn(index);
    } else for (const index of columns) this.#paintColumn(index);
    if (this.#dirtyAll || columns.size) this.#terrainContext.putImageData(this.#image, 0, 0);
    this.#dirtyAll = false;
  }

  #paintColumn(index) {
    let top = 'air';
    let height = null;
    for (const layer of this.#layers) {
      const block = this.#palette[layer.grid[index]] || 'air';
      if (['air', 'cave_air', 'void_air'].includes(shortBlock(block))) continue;
      top = block;
      height = layer.y;
      break;
    }
    this.#top[index] = top;
    this.#topY[index] = height;
    const hex = BLOCK[shortBlock(top)] || '#b0a8a0';
    const offset = index * 4;
    this.#image.data[offset] = parseInt(hex.slice(1, 3), 16);
    this.#image.data[offset + 1] = parseInt(hex.slice(3, 5), 16);
    this.#image.data[offset + 2] = parseInt(hex.slice(5, 7), 16);
    this.#image.data[offset + 3] = 255;
  }

  #draw() {
    const context = this.#context;
    this.#syncTerrain();
    context.clearRect(0, 0, this.#width, this.#height);
    context.fillStyle = this.#colors.bg;
    context.fillRect(0, 0, this.#width, this.#height);
    const [left, top] = this.#project(this.#origin, this.#origin);
    const width = this.#size * this.#scale();
    if (this.#base) {
      context.imageSmoothingEnabled = false;
      context.drawImage(this.#terrain, left, top, width, width);
    } else {
      context.strokeStyle = this.#colors.rule;
      context.lineWidth = .6;
      for (let offset = 0; offset <= this.#size; offset += 16) {
        const [horizontal, vertical] = this.#project(this.#origin + offset, this.#origin + offset);
        context.beginPath();
        context.moveTo(horizontal, top); context.lineTo(horizontal, top + width);
        context.moveTo(left, vertical); context.lineTo(left + width, vertical);
        context.stroke();
      }
    }
    this.#people = [];
    const data = this.store.data;
    if (data) {
      context.font = '600 11px system-ui, sans-serif';
      if (this.overlays.places) {
        for (const [name, x, z] of this.#placeLabels(data.meta.places)) {
          const [screenX, screenY] = this.#project(x + .5, z + .5);
          this.#label(name, screenX, screenY - 7, this.#colors.ink, 'center');
        }
      }
      if (this.overlays.agents) this.#trails();
      if (this.overlays.speech) this.#speech();
      if (this.overlays.changes) this.#highlights();
      if (this.overlays.npcs || this.overlays.routes) {
        for (const villager of data.meta.villagers) {
          const state = this.store.villagerAt(String(villager.id));
          if (!state.pos) continue;
          const [screenX, screenY] = this.#project(...state.pos);
          if (this.overlays.routes && state.travelling) {
            const destination = data.meta.places.find(place => place.name === state.destination);
            const coordinates = position(destination?.pos);
            if (coordinates) {
              context.strokeStyle = '#f08c1c'; context.lineWidth = 1; context.setLineDash([4, 4]);
              context.beginPath(); context.moveTo(screenX, screenY); context.lineTo(...this.#project(...coordinates)); context.stroke();
              context.setLineDash([]);
            }
          }
          if (this.overlays.npcs) {
            context.beginPath(); context.arc(screenX, screenY, 3.7, 0, Math.PI * 2);
            context.fillStyle = activityColor(state.activity); context.fill();
            context.strokeStyle = '#ffffff'; context.lineWidth = 1; context.stroke();
            this.#people.push({ kind: 'villager', name: villager.name || villager.id, pos: state.pos,
              state: `${humanize(state.activity)} · ${state.status}` });
          }
        }
      }
      if (this.overlays.agents) {
        for (const agent of data.meta.agents) {
          const state = this.store.agentAt(agent.name);
          if (!state.pos) continue;
          const [screenX, screenY] = this.#project(...state.pos);
          const radius = agent.name === this.store.selectedAgent ? 7 : 5.5;
          context.beginPath(); context.moveTo(screenX, screenY - radius); context.lineTo(screenX + radius, screenY);
          context.lineTo(screenX, screenY + radius); context.lineTo(screenX - radius, screenY); context.closePath();
          context.fillStyle = agentColor(agent.name); context.fill();
          context.lineWidth = agent.name === this.store.selectedAgent ? 2.5 : 1.3;
          context.strokeStyle = '#ffffff'; context.stroke();
          this.#label(agent.name, screenX + radius + 5, screenY + 4, agentColor(agent.name));
          this.#people.push({ kind: 'agent', name: agent.name, pos: state.pos,
            state: `${describe(state.skill)} · ${describe(state.state)}${state.observed == null ? ' · spawn, unobserved' : ''}` });
        }
      }
    }
    byId('zoom-label').textContent = `${Math.round(this.#zoom * 100)}%`;
    this.#tooltip();
  }

  /**
   * Place labels that don't overprint: each row of houses gets one label at its centre, and paired places
   * (the bridge's and the ford's two banks) get one label between their ends. Other places keep their own name.
   * @param {Array<object>} places scenario places with name, kind and pos
   * @returns {Array<[string, number, number]>} label text and map coordinates
   */
  #placeLabels(places) {
    const groups = new Map();
    for (const place of places) {
      const coordinates = position(place.pos);
      if (!coordinates) continue;
      let key = humanize(place.name);
      if (place.kind === 'house') key = `${place.name.startsWith('east') ? 'east' : 'west'} homes|${coordinates[1]}`;
      else if (place.kind === 'bridge' || place.kind === 'ford') key = place.kind;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(coordinates);
    }
    return [...groups].map(([key, points]) => [key.split('|')[0],
      points.reduce((sum, point) => sum + point[0], 0) / points.length,
      points.reduce((sum, point) => sum + point[1], 0) / points.length]);
  }

  #label(text, screenX, screenY, color, align = 'left') {
    const context = this.#context;
    context.textAlign = align;
    context.lineJoin = 'round';
    context.strokeStyle = this.#colors.panel;
    context.lineWidth = 3.5;
    context.strokeText(text, screenX, screenY);
    context.fillStyle = color;
    context.fillText(text, screenX, screenY);
  }

  #trails() {
    const context = this.#context;
    for (const agent of this.store.data.meta.agents) {
      const rows = this.store.data.tracks.agents[agent.name] || [];
      const end = upperBound(rows, this.store.time);
      const start = lowerBound(rows, this.store.time - 600);
      if (start >= end) continue;
      context.beginPath();
      for (let index = start; index < end; index += 1) {
        const coordinates = this.#project(rows[index][1], rows[index][2]);
        if (index === start) context.moveTo(...coordinates);
        else context.lineTo(...coordinates);
      }
      context.globalAlpha = .35; context.strokeStyle = agentColor(agent.name); context.lineWidth = 2; context.stroke(); context.globalAlpha = 1;
    }
  }

  #recentSpeech() {
    const speech = this.store.data?.speech || [];
    return speech.slice(lowerBound(speech, this.store.time - 200, event => event.t), upperBound(speech, this.store.time, event => event.t));
  }

  #speakerPosition(event) {
    const explicit = position(event.speakerPos) || position(event.pos) || position(event.speakerStartEye);
    if (explicit) return explicit;
    const key = String(event.speakerKey || '');
    if (key.startsWith('npc:')) return this.store.villagerAt(key.slice(4), event.t).pos;
    const name = key.replace(/^(?:bot|agent):/, '') || String(event.speaker || '');
    const agent = this.store.data.meta.agents.find(candidate => candidate.name === name || candidate.name === event.speaker);
    if (agent) return this.store.agentAt(agent.name, event.t).pos;
    const villager = this.store.data.meta.villagers.find(candidate => candidate.name === event.speaker || String(candidate.id) === key);
    return villager ? this.store.villagerAt(String(villager.id), event.t).pos : null;
  }

  #speech() {
    const context = this.#context;
    for (const event of this.#recentSpeech()) {
      const coordinates = this.#speakerPosition(event);
      if (!coordinates) continue;
      const radius = String(event.volume).toLowerCase() === 'shout' ? 24 : 8;
      context.beginPath(); context.arc(...this.#project(...coordinates), radius * this.#scale(), 0, Math.PI * 2);
      context.strokeStyle = '#e8b830'; context.fillStyle = '#e8b830';
      context.globalAlpha = .08; context.fill(); context.globalAlpha = .8;
      context.lineWidth = 1.4; context.stroke(); context.globalAlpha = 1;
    }
  }

  #highlights() {
    const context = this.#context;
    const rows = this.store.data.agentChanges;
    const end = upperBound(rows, this.store.time);
    const width = Math.max(3, this.#scale());
    for (let index = lowerBound(rows, this.store.time - 200); index < end; index += 1) {
      const [, worldX, , worldZ, actor] = rows[index];
      const [screenX, screenY] = this.#project(worldX + .5, worldZ + .5);
      context.fillStyle = agentColor(actor); context.globalAlpha = .45;
      context.fillRect(screenX - width / 2, screenY - width / 2, width, width); context.globalAlpha = 1;
      context.strokeStyle = '#ffd85a'; context.lineWidth = 1.3;
      context.strokeRect(screenX - width / 2, screenY - width / 2, width, width);
    }
  }

  #tooltip() {
    const tooltip = byId('map-tooltip');
    if (!this.#hover || this.#pointer || !this.store.data) { tooltip.hidden = true; return; }
    const hit = this.hitTest(...this.#hover);
    const lines = [`<strong>x ${hit.x} · z ${hit.z}${hit.y == null ? '' : ` · y ${hit.y}`}</strong>`,
      `<p>${escapeHtml(hit.block ? humanize(hit.block) : this.#base ? 'Outside the base map' : 'Base terrain unavailable')}</p>`];
    if (hit.person) lines.push(`<p><b>${escapeHtml(hit.person.name)}</b> · ${escapeHtml(hit.person.state)}</p>`);
    if (this.overlays.speech) {
      const coordinates = this.#world(...this.#hover);
      const heard = this.#recentSpeech().filter(event => {
        const speaker = this.#speakerPosition(event);
        return speaker && Math.hypot(coordinates[0] - speaker[0], coordinates[1] - speaker[1]) <= (String(event.volume).toLowerCase() === 'shout' ? 24 : 8);
      });
      for (const event of heard.slice(-3)) lines.push(`<p><b>${escapeHtml(event.speaker || event.speakerKey)}</b>: “${escapeHtml(event.text)}”</p>`);
      if (heard.length > 3) lines.push(`<p>+ ${heard.length - 3} earlier utterances in this area</p>`);
    }
    tooltip.innerHTML = lines.join('');
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(8, Math.min(this.#hover[0] + 14, this.#width - tooltip.offsetWidth - 8))}px`;
    tooltip.style.top = `${Math.max(8, Math.min(this.#hover[1] + 14, this.#height - tooltip.offsetHeight - 8))}px`;
  }

  #legend() {
    const blocks = ['water', 'grass_block', 'dirt_path', 'gravel', 'oak_slab', 'oak_planks', 'oak_leaves', 'cobblestone_wall'];
    const materials = this.#base ? new Set(this.#base.palette.map(shortBlock)) : null;
    const entries = blocks.filter(block => !materials || materials.has(block)).map(block => [humanize(block), BLOCK[block], 'swatch']);
    for (const activity of ['travelling', 'work', 'school', 'market', 'home']) entries.push([activity, activityColor(activity), 'dot']);
    byId('map-legend').innerHTML = entries.map(([label, color, shape]) => `<span><i class="${shape}" style="--marker:${color}" aria-hidden="true"></i>${escapeHtml(label)}</span>`).join('');
  }
}

/** Accessible detail tabs, time-local histories, lazy prompts and latest run reports. */
class Panels {
  #active = 'agents';
  #agentKey = '';
  #villagerKey = '';
  #eventKey = '';
  #outcomeKey = '';
  #promptKey = '';
  #promptState = 'idle';
  #promptRequest = 0;
  #promptController = null;
  #promptCache = new Map();

  /** @param {RunStore} store Shared historical run state. */
  constructor(store) {
    this.store = store;
    const names = ['agents', 'villagers', 'events', 'outcomes'];
    for (const [index, name] of names.entries()) {
      const tab = byId(`tab-${name}`);
      tab.addEventListener('click', () => this.showTab(name));
      tab.addEventListener('keydown', event => {
        let next;
        if (event.key === 'ArrowRight') next = (index + 1) % names.length;
        else if (event.key === 'ArrowLeft') next = (index + names.length - 1) % names.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = names.length - 1;
        else return;
        event.preventDefault();
        event.stopPropagation();
        this.showTab(names[next], true);
      });
    }
    byId('agent-list').addEventListener('click', event => {
      const button = event.target.closest('[data-agent]');
      if (button) store.selectAgent(button.dataset.agent);
    });
    byId('pane-agents').addEventListener('click', event => {
      const jump = event.target.closest('[data-jump]');
      if (jump) store.setTime(Number(jump.dataset.jump), true);
      if (event.target.closest('[data-retry-prompt]')) {
        this.#promptState = 'idle';
        this.#syncPrompt();
      }
    });
    byId('event-list').addEventListener('click', event => {
      const button = event.target.closest('[data-jump]');
      if (button) store.setTime(Number(button.dataset.jump), true);
    });
    byId('event-type').addEventListener('change', () => this.render());
    byId('prompt-details').addEventListener('toggle', () => this.#syncPrompt());
    store.addEventListener('data', event => {
      if (event.detail.changed) {
        this.#agentKey = '';
        this.#villagerKey = '';
        this.#eventKey = '';
        this.#outcomeKey = '';
        this.#promptCache.clear();
        this.#eventTypes();
      }
      this.render();
    });
    store.addEventListener('time', () => this.render());
    store.addEventListener('selection', () => this.showTab('agents'));
    store.addEventListener('reset', () => {
      this.#agentKey = '';
      this.#villagerKey = '';
      this.#eventKey = '';
      this.#outcomeKey = '';
      this.#cancelPrompt();
      this.#promptKey = '';
      this.#promptCache.clear();
      byId('agent-list').replaceChildren();
      byId('agent-detail').innerHTML = '<p class="empty-state">Loading agent observations…</p>';
      byId('pane-villagers').innerHTML = '<p class="empty-state">Loading villager observations…</p>';
      byId('event-list').innerHTML = '<p class="empty-state">Loading events…</p>';
      byId('pane-outcomes').innerHTML = '<p class="empty-state">Loading run report…</p>';
      byId('prompt-details').hidden = true;
      byId('prompt-content').replaceChildren();
    });
    byId('pane-villagers').innerHTML = '<p class="empty-state">Choose a run to inspect villagers.</p>';
    byId('pane-outcomes').innerHTML = '<p class="empty-state">Choose a run to inspect its report.</p>';
  }

  /** Activate a tab without rebuilding or changing it during live refresh.
   * @param {string} name Panel identifier.
   * @param {boolean} [focus] Move keyboard focus to the tab.
   */
  showTab(name, focus = false) {
    this.#active = name;
    for (const candidate of ['agents', 'villagers', 'events', 'outcomes']) {
      const selected = candidate === name;
      const tab = byId(`tab-${candidate}`);
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      byId(`pane-${candidate}`).hidden = !selected;
    }
    if (focus) byId(`tab-${name}`).focus();
    this.render();
  }

  /** Render only the active panel, reusing unchanged DOM while replay advances. */
  render() {
    if (!this.store.data) return;
    if (this.#active === 'agents') this.#agents();
    else if (this.#active === 'villagers') this.#villagers();
    else if (this.#active === 'events') this.#events();
    else if (this.#active === 'outcomes') this.#outcomes();
  }

  #agents() {
    const { data, selectedAgent, revision } = this.store;
    const agent = data.meta.agents.find(candidate => candidate.name === selectedAgent);
    if (!agent) {
      byId('agent-list').replaceChildren();
      byId('agent-detail').innerHTML = '<p class="empty-state">No agent metadata is available for this run.</p>';
      byId('prompt-details').hidden = true;
      this.#cancelPrompt();
      return;
    }
    const state = this.store.agentAt(selectedAgent);
    const history = this.store.decisionsAt(selectedAgent);
    const all = data.decisions.get(selectedAgent) || [];
    const decisionCount = upperBound(all, this.store.time, event => event.t);
    const key = `${revision}:${selectedAgent}:${state.index}:${decisionCount}`;
    if (key !== this.#agentKey) {
      const choices = byId('agent-list');
      if (choices.dataset.revision !== String(revision)) {
        choices.innerHTML = data.meta.agents.map(candidate => `<button type="button" data-agent="${escapeHtml(candidate.name)}" aria-pressed="${candidate.name === selectedAgent}"><i class="dot" style="--marker:${agentColor(candidate.name)}" aria-hidden="true"></i>${escapeHtml(candidate.name)}</button>`).join('');
        choices.dataset.revision = String(revision);
      }
      for (const button of choices.querySelectorAll('[data-agent]')) button.setAttribute('aria-pressed', String(button.dataset.agent === selectedAgent));
      const coordinates = state.pos ? `x ${Number(state.pos[0]).toFixed(1)} · z ${Number(state.pos[1]).toFixed(1)}` : 'Position unknown';
      const observation = state.observed == null ? 'Spawn estimate; no observation at or before this tick' : `Observed ${clock(state.observed)} (tick ${state.observed})`;
      const decisions = history.length ? `<div class="table-wrap"><table><thead><tr><th scope="col">When</th><th scope="col">Decision</th><th scope="col">Outcome</th></tr></thead><tbody>${history.slice().reverse().map(decision => `<tr><td class="time-cell"><button type="button" class="time-link" data-jump="${decision.t}" title="Jump to tick ${decision.t}">${escapeHtml(clock(decision.t))}</button></td><td class="decision-args"><b>${escapeHtml(decision.tool || decision.skill || 'decision')}</b><small>${escapeHtml(describe(decision.args))}</small>${decision.interrupted ? '<small>Interrupted</small>' : ''}</td><td>${escapeHtml(describe(decision.outcome))}</td></tr>`).join('')}</tbody></table></div>`
        : '<p class="muted">No decisions at or before this time.</p>';
      byId('agent-detail').className = '';
      byId('agent-detail').innerHTML = `<h3 class="agent-heading"><i class="dot" style="--marker:${agentColor(agent.name)}" aria-hidden="true"></i>${escapeHtml(agent.name)}</h3>
        <p class="agent-state">${escapeHtml(coordinates)}<br>${escapeHtml(describe(state.skill))} · ${escapeHtml(describe(state.state))}<br>${escapeHtml(observation)}</p>
        <dl class="goals"><div><dt>Public goal</dt><dd>${escapeHtml(agent.public_goal || 'Not recorded')}</dd></div><div><dt>Private goal</dt><dd>${escapeHtml(agent.private_goal || 'Not recorded')}</dd></div></dl>
        <div class="subheading"><h3>Last 10 decisions</h3><span class="muted">${decisionCount.toLocaleString()} so far · includes waits</span></div>${decisions}`;
      this.#agentKey = key;
    }
    this.#syncPrompt();
  }

  #cancelPrompt() {
    this.#promptController?.abort();
    this.#promptController = null;
    this.#promptRequest += 1;
    this.#promptState = 'idle';
  }

  #syncPrompt() {
    const store = this.store;
    const details = byId('prompt-details');
    if (!store.data || !store.selectedAgent || this.#active !== 'agents') return;
    const latest = store.decisionsAt(store.selectedAgent).at(-1);
    const key = `${store.generation}:${store.revision}:${store.selectedAgent}:${latest?.t ?? 'none'}`;
    details.hidden = false;
    byId('prompt-time').textContent = latest ? `· ${clock(latest.t)}` : '· no decision yet';
    if (key !== this.#promptKey) {
      this.#cancelPrompt();
      this.#promptKey = key;
      byId('prompt-content').innerHTML = `<p class="muted">${latest ? 'Open to load what this agent saw at its latest decision at or before the selected time.' : 'No decision exists at or before this time.'}</p>`;
    }
    if (!details.open || !latest || this.#promptState !== 'idle') return;
    if (this.#promptCache.has(key)) {
      this.#showPrompt(this.#promptCache.get(key));
      this.#promptState = 'loaded';
      return;
    }
    const generation = store.generation;
    const name = store.name;
    const variant = store.variant;
    const agent = store.selectedAgent;
    const time = store.time;
    const request = ++this.#promptRequest;
    this.#promptController = new AbortController();
    this.#promptState = 'loading';
    byId('prompt-content').innerHTML = '<p class="muted" role="status">Loading decision prompt…</p>';
    store.api.run(name, 'prompt', variant, { agent, t: time }, this.#promptController.signal).then(result => {
      if (request !== this.#promptRequest || key !== this.#promptKey || generation !== store.generation) return;
      if (result?.decision?.t > store.time) {
        this.#promptState = 'error';
        byId('prompt-content').innerHTML = '<p class="muted">No time-safe prompt is available at this tick. Refresh the run and try again.</p>';
        return;
      }
      this.#promptCache.set(key, result);
      if (this.#promptCache.size > 24) this.#promptCache.delete(this.#promptCache.keys().next().value);
      this.#promptState = 'loaded';
      this.#showPrompt(result);
    }).catch(error => {
      if (request !== this.#promptRequest || key !== this.#promptKey || generation !== store.generation) return;
      this.#promptState = 'error';
      byId('prompt-content').innerHTML = `<p class="muted">Could not load the prompt: ${escapeHtml(error.message)}</p><button type="button" data-retry-prompt>Retry prompt</button>`;
    });
  }

  #showPrompt(result) {
    if (result == null || result.prompt == null) {
      byId('prompt-content').innerHTML = '<p class="muted">No prompt was recorded for this decision.</p>';
      return;
    }
    const turns = (result.history || []).map(turn => `<details class="turn"><summary>${escapeHtml(`${clock(turn.t)} · ${turn.tool ?? 'no action'} → ${turn.result}`)}</summary><pre tabindex="0">${escapeHtml(describe(turn.message))}</pre><pre class="action">${escapeHtml(`${turn.tool ?? '(no action)'} ${JSON.stringify(turn.args ?? {})}`)}</pre></details>`).join('');
    const earlier = turns ? `<p class="muted">Earlier in this conversation (${result.history.length} turn${result.history.length === 1 ? '' : 's'}, oldest first):</p>${turns}<p class="muted">This decision's message:</p>` : '';
    byId('prompt-content').innerHTML = `<p class="muted">${escapeHtml(result.decision?.t != null ? `Decision at ${clock(result.decision.t)} · tick ${result.decision.t}` : 'Latest decision at or before the selected time')}</p>${earlier}<pre tabindex="0">${escapeHtml(describe(result.prompt))}</pre>`;
  }

  #villagers() {
    const { data, revision } = this.store;
    const people = data.meta.villagers.map(villager => ({ villager, state: this.store.villagerAt(String(villager.id)) }));
    const key = `${revision}:${people.map(({ state }) => `${state.index},${state.tripIndex},${state.arrived}`).join(';')}`;
    if (key === this.#villagerKey) return;
    this.#villagerKey = key;
    if (!people.length) {
      byId('pane-villagers').innerHTML = '<p class="empty-state">No villager metadata is available for this run.</p>';
      return;
    }
    const rows = people.map(({ villager, state }) => {
      const trip = state.trip;
      const onTime = state.arrived ? state.onTime === true ? '<span class="pill good">On time</span>'
        : state.onTime === false ? '<span class="pill bad">Late</span>' : '<span class="pill pending">Timing unknown</span>'
        : trip && trip[1] == null ? '<span class="pill bad">Failed · no arrival</span>' : '<span class="pill pending">Not known yet</span>';
      const coordinates = state.pos ? `x ${Number(state.pos[0]).toFixed(1)}, z ${Number(state.pos[1]).toFixed(1)}` : 'Position unknown';
      const tripInfo = !trip ? `Home: ${humanize(villager.home || 'unknown')}`
        : `${state.travelling ? 'To' : state.arrived ? 'Last destination:' : 'Attempted:'} ${humanize(state.destination)} · departed ${clock(trip[0])}${trip[6] ? ` from ${humanize(trip[6])}` : ''}`;
      return `<tr><td><span class="villager-name">${escapeHtml(villager.name || villager.id)}</span><small>${escapeHtml(villager.role || 'villager')} · ${escapeHtml(villager.id)}</small><small>${escapeHtml(coordinates)}${state.observed == null ? ' (estimate)' : ''}</small></td>
        <td><span class="activity"><i class="dot" style="--marker:${activityColor(state.activity)}" aria-hidden="true"></i>${escapeHtml(humanize(state.activity))}</span><small>${escapeHtml(state.status)}</small></td>
        <td>${escapeHtml(tripInfo)}${state.arrived ? `<small>Arrived ${escapeHtml(clock(trip[1]))}</small>` : ''}</td>
        <td>${onTime}${state.outcome != null ? `<small>${escapeHtml(describe(state.outcome))}</small>` : ''}</td></tr>`;
    }).join('');
    byId('pane-villagers').innerHTML = `<p class="muted">${people.length} villagers. Activity is inferred from recorded trips, not directly observed. Live journeys may not be logged until finished. Null-arrival trips failed; they are not ongoing travel.</p>
      <div class="table-wrap"><table><thead><tr><th scope="col">Villager / position</th><th scope="col">Activity</th><th scope="col">Current / last trip</th><th scope="col">Arrival status</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  #eventTypes() {
    const picker = byId('event-type');
    const selected = picker.value;
    const types = [...new Set(this.store.data.events.map(event => event.type))].sort();
    const markup = '<option value="">All types</option>' + types.map(type => `<option value="${escapeHtml(type)}">${escapeHtml(humanize(type))}</option>`).join('');
    picker.innerHTML = markup;
    picker.value = types.includes(selected) ? selected : '';
  }

  #events() {
    const { events } = this.store.data;
    const time = this.store.time;
    const start = lowerBound(events, time - 1200, event => event.t);
    const end = upperBound(events, time + 1200, event => event.t);
    const now = upperBound(events, time, event => event.t);
    const filter = byId('event-type').value;
    const key = `${this.store.revision}:${start}:${end}:${now}:${filter}`;
    if (key === this.#eventKey) return;
    this.#eventKey = key;
    const windowEvents = events.slice(start, end).filter(event => !filter || event.type === filter);
    byId('event-list').className = '';
    byId('event-list').innerHTML = windowEvents.length ? `<p class="muted">${windowEvents.length.toLocaleString()} events in this window. Dashed entries are after the selected time.</p><ol class="event-list">${windowEvents.map(event => {
      const content = this.#describeEvent(event);
      return `<li class="${event.t > time ? 'event-future' : ''}"><button type="button" class="event-jump" data-jump="${event.t}" aria-label="Jump to ${escapeHtml(humanize(event.type))}, tick ${event.t}"><span class="event-meta"><span>${escapeHtml(clock(event.t))} · ${event.t > time ? 'upcoming' : `tick ${event.t}`}</span><span class="event-kind">${escapeHtml(humanize(event.type))}</span></span><span class="event-description">${escapeHtml(content.description)}</span>${content.detail ? `<span class="muted">${escapeHtml(content.detail)}</span>` : ''}</button></li>`;
    }).join('')}</ol>` : '<p class="empty-state">No matching events within ±1,200 ticks of the selected time.</p>';
  }

  #describeEvent(event) {
    const who = event.agent || event.bot || event.buyer || event.speaker || event.speakerKey || event.actor || '';
    if (event.type === 'decision') return { description: `${who}: ${event.tool || 'decision'} ${event.args == null ? '' : describe(event.args)}`,
      detail: `Outcome: ${describe(event.outcome)}${event.interrupted ? ' · interrupted' : ''}` };
    if (event.type === 'utterance') return { description: `${who}: “${event.text || ''}”`,
      detail: `${event.volume || 'talk'} · ${Array.isArray(event.listeners) ? `Listeners (${event.listeners.length}): ${event.listeners.map(describe).join(', ') || 'none'}` : 'Listeners not recorded'}` };
    if (event.type === 'transfer') return { description: `${describe(event.from)} → ${describe(event.to)}: ${event.count ?? ''} ${humanize(event.item || 'item')}`, detail: '' };
    if (event.type === 'purchase') return { description: `${who} bought ${event.count ?? ''} ${humanize(event.item || 'item')} at ${describe(event.shop)}`, detail: `Cost: ${describe(event.cost)}` };
    if (event.type === 'skill_end') return { description: `${who}: ${describe(event.skill)} ended`, detail: `${describe(event.outcome)}${event.reason ? ` · ${event.reason}` : ''}` };
    const fields = Object.fromEntries(Object.entries(event).filter(([key]) => !['t', 'type', 'user_prompt', 'prompt'].includes(key)));
    return { description: describe(fields), detail: '' };
  }

  #outcomes() {
    const { meta } = this.store.data;
    const report = meta.report;
    const firstTrue = report?.first_true || {};
    const key = `${this.store.revision}:${JSON.stringify(report ?? null)}:${Object.values(firstTrue).map(time => typeof time === 'number' && time <= this.store.time).join(',')}`;
    if (key === this.#outcomeKey) return;
    this.#outcomeKey = key;
    if (!report || !Object.keys(report).length) {
      byId('pane-outcomes').innerHTML = '<p class="empty-state">No report has been written for this run yet. Map replay, decisions and events are still available.</p>';
      return;
    }
    const outcomeRows = Object.entries(report.outcomes || {}).map(([name, outcome]) => `<tr><th scope="row">${escapeHtml(humanize(name))}</th><td>${typeof outcome === 'boolean' || outcome == null ? pill(outcome) : escapeHtml(describe(outcome))}</td></tr>`).join('');
    const milestones = Object.entries(firstTrue).map(([name, time]) => `<tr><th scope="row">${escapeHtml(humanize(name))}</th><td>${typeof time === 'number' ? `${escapeHtml(clock(time))}<small>${time <= this.store.time ? 'Reached by the selected time' : 'After the selected time — not yet reached here'}</small>` : 'Not recorded'}</td></tr>`).join('');
    const welfare = Object.entries(report.welfare || {}).map(([day, value]) => `<tr><th scope="row">${escapeHtml(day)}</th><td>${escapeHtml(typeof value === 'number' ? value.toFixed(3) : describe(value))}</td></tr>`).join('');
    const stats = Object.entries(report.stats || {}).map(([name, value]) => `<tr><th scope="row">${escapeHtml(humanize(name))}</th><td>${escapeHtml(describe(value))}</td></tr>`).join('');
    const table = rows => `<div class="table-wrap"><table><tbody>${rows}</tbody></table></div>`;
    byId('pane-outcomes').innerHTML = `<p class="report-note">Latest run report, not a historical snapshot. Outcomes, welfare and totals may include information after the selected tick. Only milestone timing is compared with the clock.</p>
      ${outcomeRows ? `<div class="subheading"><h3>Role outcomes</h3></div>${table(outcomeRows)}` : '<p class="muted">No role outcomes recorded.</p>'}
      ${milestones ? `<div class="subheading"><h3>First reached</h3></div>${table(milestones)}` : ''}
      ${welfare ? `<div class="subheading"><h3>Villager welfare</h3></div>${table(welfare)}` : ''}
      ${stats ? `<div class="subheading"><h3>Run statistics</h3></div>${table(stats)}` : ''}
      <details class="report-raw"><summary>Complete report</summary><pre tabindex="0">${escapeHtml(JSON.stringify(report, null, 2))}</pre></details>`;
  }
}

const api = new Api();
const store = new RunStore(api);
const mapView = new MapView(store);
const timeline = new Timeline(store);
const panels = new Panels(store);
let runs = [];
let listing = false;
const initialQuery = new URLSearchParams(window.location.search);
const initialVariant = initialQuery.get('variant');
if (['sabotage', 'accident'].includes(initialVariant)) byId('variant-picker').value = initialVariant;

function updateStatus() {
  const status = byId('status');
  const meta = store.data?.meta;
  status.classList.toggle('has-error', Boolean(store.error));
  if (store.error) status.textContent = store.error;
  else if (store.loading) status.textContent = `Loading ${store.name}…`;
  else if (meta) status.textContent = `${store.name} · ${meta.variant || 'unknown scenario'} · ${store.follow ? 'Following latest tick' : meta.live ? 'Live run; replay time is yours to explore' : 'Recorded run'} · updated ${store.updatedAt.toLocaleTimeString()}`;
  else status.textContent = runs.length ? 'Select a run to begin.' : 'No runs found. Point the viewer server at a directory containing run logs; checking again every 5 seconds.';
  const badge = byId('run-badge');
  badge.classList.toggle('live', Boolean(meta?.live));
  badge.textContent = meta ? meta.live ? 'Live run' : 'Recorded run' : store.loading ? 'Loading' : 'No run selected';
  byId('variant-picker').options[0].textContent = meta ? `Auto (${runs.find(run => run.name === store.name)?.variant || 'sabotage'})` : 'Auto-detect';
  const warnings = meta?.warnings || [];
  byId('warnings').hidden = warnings.length === 0;
  byId('warning-summary').textContent = `${warnings.length} run warning${warnings.length === 1 ? '' : 's'} · missing sources may limit observations`;
  byId('warning-list').innerHTML = warnings.map(warning => `<li>${escapeHtml(warning)}</li>`).join('');
}

async function refreshRuns() {
  if (listing) return;
  listing = true;
  try {
    const response = await api.get('/api/runs');
    const modified = run => typeof run.modified === 'number' ? run.modified : number(Date.parse(run.modified), 0);
    runs = (Array.isArray(response) ? response : []).slice().sort((first, second) => modified(second) - modified(first) || number(second.end) - number(first.end) || String(second.name).localeCompare(String(first.name)));
    const picker = byId('run-picker');
    const options = runs.map(run => `<option value="${escapeHtml(run.name)}">${escapeHtml(run.name)}${run.live ? ' · LIVE' : ''}</option>`);
    if (store.name && !runs.some(run => run.name === store.name)) options.push(`<option value="${escapeHtml(store.name)}">${escapeHtml(store.name)} · not listed</option>`);
    const markup = options.join('') || '<option value="">No runs found</option>';
    if (picker.innerHTML !== markup) picker.innerHTML = markup;
    picker.disabled = options.length === 0;
    if (store.name) {
      picker.value = store.name;
      const selected = runs.find(run => run.name === store.name);
      const meta = store.data?.meta;
      if (selected && meta && (selected.modified !== meta.modified || selected.live !== meta.live
        || selected.end !== meta.end)) void store.refresh(true);
    }
    else if (runs.length) {
      const requested = initialQuery.get('run');
      const name = runs.some(run => run.name === requested) ? requested : runs[0].name;
      picker.value = name;
      void store.load(name, byId('variant-picker').value);
    }
    byId('index-error').hidden = true;
    updateStatus();
  } catch (error) {
    byId('index-error').textContent = `Could not refresh the run list: ${error.message}. Start the local viewer server if needed. Retrying every 5 seconds.`;
    byId('index-error').hidden = false;
    if (!store.name) {
      byId('run-picker').innerHTML = '<option value="">Server unavailable</option>';
      byId('status').textContent = 'Waiting for the local viewer API. No external services are used.';
    }
  } finally {
    listing = false;
  }
}

byId('run-picker').addEventListener('change', event => {
  if (event.target.value) void store.load(event.target.value, byId('variant-picker').value);
});
byId('variant-picker').addEventListener('change', event => {
  if (store.name) void store.load(store.name, event.target.value);
});
byId('refresh').addEventListener('click', () => { void refreshRuns(); void store.refresh(true); });
for (const event of ['status', 'data', 'reset', 'follow']) store.addEventListener(event, updateStatus);
void refreshRuns();
setInterval(() => { void refreshRuns(); void store.refresh(); }, 5000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) { void refreshRuns(); void store.refresh(); }
});
