const fs = require('fs');
const readline = require('readline');

// get path from arg
const path = process.argv[2];
const n = parseInt(process.argv[3]);

const best = [];

const rl = readline.createInterface({
    input: fs.createReadStream(path),
    crlfDelay: Infinity
});

let lineCount = 0;

rl.on('line', (line) => {
    const parts = line.split(',');
    if (parts.length !== 8) {
        return;
    }
    const tallyCorrect = parseInt(parts[parts.length - 1]);
    best.push({ line, tallyCorrect });
    best.sort((a, b) => b.tallyCorrect - a.tallyCorrect);
    if (best.length > n) {
        best.pop();
    }
    lineCount++;
    if (lineCount % 10000000 === 0) {
        console.log(`Processed ${lineCount} lines`);
        console.log(best.map(x => x.line).join('\n'));
    }
});

rl.on('close', () => {
    console.log('Finished');
    console.log(best.map(x => x.line).join('\n'));
});