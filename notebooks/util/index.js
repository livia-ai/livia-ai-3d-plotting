import fs from 'fs';

const FILES = [
  '../triplets_01.json',
  '../triplets_02.json'
];

const MAX_LENGTH = 1000;

const data = FILES.reduce((acc, file) => {
  console.log(file);
  const json = fs.readFileSync(file, 'utf8');
  const parsed = JSON.parse(json);

  const filtered = parsed.filter(triplet => {
    // Filter out any triplet with 'null' image URLs
    const { anchor, similar, dissimilar } = triplet;

    const isValid = t => !t.image.includes('null');

    return [anchor, similar, dissimilar].every(isValid);
  });

  return [...acc, ...filtered];
}, []);

// Truncate to MAX_LENGTH
const truncated = data.slice(0, MAX_LENGTH);

// Write data to JSON file
fs.writeFileSync('../triplets_filtered.json', JSON.stringify(truncated));
console.log(`Done. ${truncated.length} valid triplets`);
