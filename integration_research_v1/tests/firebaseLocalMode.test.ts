import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

test('local demo mode disables Firebase and starts with users but no scan records', async () => {
  const previous = process.env.USE_LOCAL_DB;
  const previousDataDir = process.env.LOCAL_DATA_DIR;
  const dataDir = mkdtempSync(join(tmpdir(), 'nafld-local-db-test-'));
  process.env.USE_LOCAL_DB = 'true';
  process.env.LOCAL_DATA_DIR = dataDir;

  try {
    const firebase = await import('../../src/lib/firebase.ts');
    assert.equal(firebase.isFirebaseConfigured(), false);
    assert.equal(firebase.db, null);
    assert.equal(firebase.storage, null);

    const localDb = await import('../../src/lib/db.ts');
    assert.deepEqual(await localDb.dbGetScans(), []);
    assert.equal((await localDb.dbGetUserByEmailWithPassword('doctor@system.com'))?.role, 'doctor');
    assert.equal((await localDb.dbGetUserByEmailWithPassword('admin@system.com'))?.role, 'admin');
  } finally {
    rmSync(dataDir, { recursive: true, force: true });
    if (previous === undefined) {
      delete process.env.USE_LOCAL_DB;
    } else {
      process.env.USE_LOCAL_DB = previous;
    }
    if (previousDataDir === undefined) {
      delete process.env.LOCAL_DATA_DIR;
    } else {
      process.env.LOCAL_DATA_DIR = previousDataDir;
    }
  }
});
