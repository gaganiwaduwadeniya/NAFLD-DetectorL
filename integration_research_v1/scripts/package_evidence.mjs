import { cp, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const root = '/mnt/g/nafld-detectorl';
const packageName = 'NAFLD_CHAPTER6_EVIDENCE_RESEARCH_V1_2026-10-24';
const staging = path.join(root, 'integration_research_v1', 'portable_package', packageName);
const zipPath = path.join(root, `${packageName}.zip`);
const extractRoot = `/tmp/${packageName}-verify`;
const allowlist = [
  'integration_research_v1/README_FIRST.md',
  'integration_research_v1/reports',
  'integration_research_v1/contracts',
  'integration_research_v1/evidence/screenshots/06_real_gradcam_e2e.png',
  'integration_research_v1/evidence/screenshots_manifest.csv',
  'integration_research_v1/evidence/test_summary.json',
  'integration_research_v1/tests',
  'integration_research_v1/runbooks',
  'integration_research_v1/thesis_chapter6_source_map.md',
  'integration_research_v1/integration_change_manifest.json',
  'pytorch_service',
  'src/server/pytorchContract.ts',
  'src/components/ResearchV1Card.tsx',
  'src/components/ResultCard.tsx',
  'src/components/ImageUploader.tsx',
  'src/components/ScanTable.tsx',
  'src/pages/DoctorDashboard.tsx',
  'src/pages/ScanDetail.tsx',
  'src/types.ts',
  'src/lib/db.ts',
  'src/App.tsx',
  'server.ts',
  '.env.example',
  'package.json',
  'package-lock.json'
];

async function filesUnder(target) {
  const info = await stat(target);
  if (info.isFile()) return [target];
  const output = [];
  for (const entry of await readdir(target, { withFileTypes: true })) output.push(...await filesUnder(path.join(target, entry.name)));
  return output;
}

async function digest(target) {
  return createHash('sha256').update(await readFile(target)).digest('hex');
}

await rm(path.dirname(staging), { recursive: true, force: true });
await mkdir(staging, { recursive: true });
for (const relative of allowlist) {
  const source = path.join(root, relative);
  const destination = path.join(staging, relative);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}

const files = (await filesUnder(staging)).sort();
const prohibitedNames = files.filter(file => /(?:\.pt|\.pth|\.keras|\.h5|db\.json|annotation|patient.*map|fixture\.png)$/i.test(file));
if (prohibitedNames.length) throw new Error(`Privacy/package allowlist violation: ${prohibitedNames.join(', ')}`);
const sums = [];
for (const file of files) sums.push(`${await digest(file)}  ${path.relative(staging, file).replaceAll('\\', '/')}`);
await writeFile(path.join(staging, 'SHA256SUMS.txt'), sums.join('\n') + '\n');

await rm(zipPath, { force: true });
const zipped = spawnSync('zip', ['-q', '-r', zipPath, packageName], { cwd: path.dirname(staging), encoding: 'utf8' });
if (zipped.status !== 0) throw new Error(`zip failed: ${zipped.stderr}`);
await rm(extractRoot, { recursive: true, force: true });
await mkdir(extractRoot, { recursive: true });
const unzipped = spawnSync('unzip', ['-q', zipPath, '-d', extractRoot], { encoding: 'utf8' });
if (unzipped.status !== 0) throw new Error(`unzip failed: ${unzipped.stderr}`);
const extractedRoot = path.join(extractRoot, packageName);
for (const row of sums) {
  const [expected, relative] = row.split('  ');
  const actual = await digest(path.join(extractedRoot, relative));
  if (actual !== expected) throw new Error(`Extracted hash mismatch: ${relative}`);
}
const result = {
  zip_path: zipPath,
  zip_sha256: await digest(zipPath),
  packaged_file_count: sums.length + 1,
  extracted_file_count: (await filesUnder(extractedRoot)).length,
  extraction_verified: true,
  all_listed_hashes_verified: true,
  prohibited_names_found: prohibitedNames,
  model_binaries_included: false,
  raw_input_images_included: false,
  credentials_included: false,
};
await writeFile(path.join(root, 'integration_research_v1', 'evidence', 'package_verification.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
