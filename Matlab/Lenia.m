%function LeniaMatlab()
    global key ZIP_START KERNEL_IDS DELTA_IDS
    key = '';
    ZIP_START = 192;

    KERNEL_IDS = {'quad4', 'bump4'};
    DELTA_IDS = {'quad4', 'gaus'};

    SIZE = pow2(8);
    MID = floor(SIZE / 2);
    RAND_BORDER = floor(SIZE / 6);

    world = zeros(SIZE, SIZE);
    potential = zeros(SIZE, SIZE);
    delta = -ones(SIZE, SIZE);

    tick = 0;
    gen = 0;
    isRun = true;
    isCalc = true;
    isCalcOnce = false;
    isNewWorld = true;
    isNewSec = false;
    isRedraw = false;
    isCalcStat = false;
    isRotateStat = false;
    isAutoCenter = false;
    isLastRandom = false;
    calcNum = 0;
    statIdx = 1;
    lastIdx = 1;
    stftIdx = 0;
    seq = 1;
    uiMode = 2;
    mainPanel = 1;

    statNum = 10;
    stftNum = statNum - 5;
    stat = NaN(1, statNum);
    stft = [];

    fig = figure('Position',[0 0 650 650], 'NumberTitle','off', 'Name','LeniaMatlab');
    panels = struct();
    [fig, panels] = SetUIMode(uiMode, mainPanel, fig, panels, SIZE);

    %R = 26; peaks = [1]; mu = 0.15; sigma = 0.014; dt = 0.1;
    %R = 36; peaks = [1 1 1]; mu = 0.25; sigma = 0.035; dt = 0.1;
    %world = RandomWorld(world, RAND_BORDER, 0.5);
    [name, R, peaks, mu, sigma, dt, deltaType, kernelType, cells] = LoadCells('0');
    if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
    [cells, R] = DoubleCells(cells, R);
    world = AddCells(world, cells);

    [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);

    set(fig, 'KeyPressFcn',@KeyPress)  %WindowKeyPressFcn
    menus = AddCustomMenu(fig);

    colormap(jet);
    if isOctave; graphics_toolkit("fltk"); end
    %available_graphics_toolkits, loaded_graphics_toolkits, register_graphics_toolkit("fltk")

    while isRun && isvalid(fig)
        switch key
            case 'return'
                isCalc = ~isCalc;
            case 'space'
                isCalc = false;
                isCalcOnce = true;
            case 'escape'
                isRun = false;

            case {'backspace','delete'}
                world = zeros(SIZE, SIZE);
                isNewWorld = true;
            case 'n'
                world = zeros(SIZE, SIZE);
                world = RandomPatches(world, R, RAND_BORDER);
                isNewWorld = true;

            case 'b' %not in menu
                mu = randi(40) / 100 + 0.1;
                sigma = mu * 0.12;
                peaks = zeros(1,randi(4)+1);
                for i=1:length(peaks)
                    peaks(i) = randi(12)/12;
                end
                peaks(randi(length(peaks))) = 1;
                if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);

                world = zeros(SIZE, SIZE);
                world = RandomPatches(world, R, RAND_BORDER);
                isNewWorld = true;
                isLastRandom = true;
                
            case 'z'
                world = zeros(SIZE, SIZE);
                world = AddCells(world, cells);
                isNewSec = true;
            case 'x'
                world = AddCells(world, cells);
                isNewSec = true;
            case {'1','2','3','4','5','6','7','8','9','0','S+1','S+2','S+3','S+4','S+5','S+6','S+7','S+8','S+9','S+0'}
                [name, R, peaks, mu, sigma, dt, deltaType, kernelType, cells] = LoadCells(key);
                if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);
                world = zeros(SIZE, SIZE);
                world = AddCells(world, cells);
                isNewWorld = true;

            %{
            case 'r'
                [cells, R] = DoubleCells(cells, R);
                if size(cells,1) <= SIZE && size(cells,2) <= SIZE
                    [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);
                    world = zeros(SIZE, SIZE);
                    world = AddCells(world, cells);
                    isNewWorld = true;
                end
            case 'f'
                [cells, R] = HalfCells(cells, R);
                if size(cells,1) >= 2 && size(cells,2) >= 2
                    [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);
                    world = zeros(SIZE, SIZE);
                    world = AddCells(world, cells);
                    isNewWorld = true;
                end
            %}
            case 'leftarrow';    world = TransformWorld(world, R, -10, 0, 1, 0, 0, false); isRedraw = true;
            case 'rightarrow';   world = TransformWorld(world, R, +10, 0, 1, 0, 0, false); isRedraw = true;
            case 'uparrow';      world = TransformWorld(world, R, 0, -10, 1, 0, 0, false); isRedraw = true;
            case 'downarrow';    world = TransformWorld(world, R, 0, +10, 1, 0, 0, false); isRedraw = true;
            case 'S+leftarrow';  world = TransformWorld(world, R, -1, 0, 1, 0, 0, false); isRedraw = true;
            case 'S+rightarrow'; world = TransformWorld(world, R, +1, 0, 1, 0, 0, false); isRedraw = true;
            case 'S+uparrow';    world = TransformWorld(world, R, 0, -1, 1, 0, 0, false); isRedraw = true;
            case 'S+downarrow';  world = TransformWorld(world, R, 0, +1, 1, 0, 0, false); isRedraw = true;

            case 'home';         world = TransformWorld(world, R, 0, 0, 1, -45, 0, ~isCalc); isRedraw = true;
            case 'end';          world = TransformWorld(world, R, 0, 0, 1, +45, 0, ~isCalc); isRedraw = true;
            case 'S+home';       world = TransformWorld(world, R, 0, 0, 1, -1, 0, ~isCalc); isRedraw = true;
            case 'S+end';        world = TransformWorld(world, R, 0, 0, 1, +1, 0, ~isCalc); isRedraw = true;

            case 'pageup';       world = TransformWorld(world, R, 0, 0, 1, 0, 1, false); isRedraw = true;
            case 'pagedown';     world = TransformWorld(world, R, 0, 0, 1, 0, 2, false); isRedraw = true;
            case 'S+pageup';     world = TransformWorld(world, R, 0, 0, 1, 0, 4, false); isNewWorld = true;
            case 'S+pagedown';   world = TransformWorld(world, R, 0, 0, 1, 0, 5, false); isNewWorld = true;

            case 'r';   [world, R2] = TransformWorld(world, R, 0, 0, 2,   0, 0, ~isCalc); if R ~= R2; R = R2; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewWorld = true; end
            case 'f';   [world, R2] = TransformWorld(world, R, 0, 0, 1/2, 0, 0, ~isCalc); if R ~= R2; R = R2; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewWorld = true; end
            case 'A+r'; [world, R2] = TransformWorld(world, R, 0, 0, 3/2, 0, 0, ~isCalc); if R ~= R2; R = R2; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewWorld = true; end
            case 'A+f'; [world, R2] = TransformWorld(world, R, 0, 0, 2/3, 0, 0, ~isCalc); if R ~= R2; R = R2; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewWorld = true; end
            case 'S+r'; [world, R2] = TransformWorld(world, R, 0, 0, (R+1)/R, 0, 0, ~isCalc); if R ~= R2; R = R2; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewWorld = true; end
            case 'S+f'; [world, R2] = TransformWorld(world, R, 0, 0, (R-1)/R, 0, 0, ~isCalc); if R ~= R2; R = R2; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewWorld = true; end

            case {'M+c', 'C+c'}
                clipboard('copy', [ Rule2Str(R, peaks, mu, sigma, dt, deltaType, kernelType) ';cells=' World2Str(world, m) ]);
            case {'M+v', 'C+v'}
                st = clipboard('paste');
                if startsWith(st, 'R=')
                    [R, peaks, mu, sigma, dt, deltaType, kernelType, cells] = Str2RuleAndCell(st);
                    if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                    [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);
                    world = AddCells(world, cells);
                end
                name = '(pasted)';
                isNewSec = true;
            case {'M+s', 'C+s'}
                csvwrite('cells.csv', world);
                isNewWorld = true;
            case {'M+l', 'C+l'}
                world = csvread('cells.csv');

            case 'q'; mu = mu + 0.01; if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end; isNewSec = true;
            case 'a'; mu = mu - 0.01; if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end; isNewSec = true;
            case 'w'; sigma = sigma + 0.001; isNewSec = true;
            case 's'; sigma = sigma - 0.001; isNewSec = true;
            case 'S+q'; mu = mu + 0.001; if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end; isNewSec = true;
            case 'S+a'; mu = mu - 0.001; if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end; isNewSec = true;
            case 'S+w'; sigma = sigma + 0.0001; isNewSec = true;
            case 'S+s'; sigma = sigma - 0.0001; isNewSec = true;
            case 'S+t'; dt = 0.01; isNewSec = true;
            case 't'; dt = 0.03; isNewSec = true;
            case 'g'; dt = 0.1; isNewSec = true;
            case 'S+g'; dt = 0.2; isNewSec = true;
                
            case 'leftbracket'
                if length(peaks) > 1
                    peaks = peaks(1:end-1);
                    [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); 
                    isNewSec = true;
                end
            case 'rightbracket'
                if length(peaks) < 5
                    peaks = [peaks 1];
                    [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); 
                    isNewSec = true;
                end
            case 'y'; if length(peaks) >= 2 && round(peaks(1)*12) < 12; peaks(1) = (round(peaks(1)*12)+1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'u'; if length(peaks) >= 2 && round(peaks(2)*12) < 12; peaks(2) = (round(peaks(2)*12)+1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'i'; if length(peaks) >= 3 && round(peaks(3)*12) < 12; peaks(3) = (round(peaks(3)*12)+1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'o'; if length(peaks) >= 4 && round(peaks(4)*12) < 12; peaks(4) = (round(peaks(4)*12)+1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'p'; if length(peaks) >= 5 && round(peaks(5)*12) < 12; peaks(5) = (round(peaks(5)*12)+1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'S+y'; if length(peaks) >= 2 && round(peaks(1)*12) > 0; peaks(1) = (round(peaks(1)*12)-1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'S+u'; if length(peaks) >= 2 && round(peaks(2)*12) > 0; peaks(2) = (round(peaks(2)*12)-1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'S+i'; if length(peaks) >= 3 && round(peaks(3)*12) > 0; peaks(3) = (round(peaks(3)*12)-1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'S+o'; if length(peaks) >= 4 && round(peaks(4)*12) > 0; peaks(4) = (round(peaks(4)*12)-1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end
            case 'S+p'; if length(peaks) >= 5 && round(peaks(5)*12) > 0; peaks(5) = (round(peaks(5)*12)-1)/12; [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType); isNewSec = true; end

            %not in menu
            case 'A+q'; sigma = lastSigma; mu = mu + 0.001; if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end; world = eval(clipboard('paste')); statIdx = lastIdx; stat(statIdx+1:end, :) = []; stftIdx = 0; stft = []; isNewSec = true; 
            case 'A+a'; sigma = lastSigma; mu = mu - 0.001; if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end; world = eval(clipboard('paste')); statIdx = lastIdx; stat(statIdx+1:end, :) = []; stftIdx = 0; stft = []; isNewSec = true;
            case 'A+w'; lastSigma = sigma; sigma = sigma + 0.0001; clipboard('copy', mat2str(world)); isNewSec = true;
            case 'A+s'; lastSigma = sigma; sigma = sigma - 0.0001; clipboard('copy', mat2str(world)); isNewSec = true;

            case 'm'
                isAutoCenter = ~isAutoCenter;
                world = CenterWorld(world, m);
                isRedraw = true;
            case 'S+m'
                world = CenterWorld(world, m);
                isRedraw = true;
            case 'j'
                if uiMode == 3
                    isCalcStat = ~isCalcStat;
                else
                    uiMode = min(uiMode + 1, 3);
                    [fig, panels] = SetUIMode(uiMode, mainPanel, fig, panels, SIZE);
                    if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                    if uiMode == 3
                        isCalcStat = true;
                    end
                    isRedraw = true;
                end
            case 'S+j'
                if uiMode == 1
                    mainPanel = mod(mainPanel, 3) + 1;
                else
                    uiMode = max(uiMode - 1, 1);
                    mainPanel = 1;
                end
                [fig, panels] = SetUIMode(uiMode, mainPanel, fig, panels, SIZE);
                if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                isCalcStat = false;
                isRedraw = true;
            case 'k'
                statIdx = lastIdx;
                if statIdx > 1
                    stat(statIdx+1:end, :) = [];
                else
                    stat = NaN(1, statNum);
                end
                stftIdx = 0;
                stft = [];
                isRedraw = true;
            case 'S+k'
                statIdx = 1;
                lastIdx = 1;
                stat = NaN(1, statNum);
                stftIdx = 0;
                stft = [];
                isRedraw = true;
            case 'l'
                isRotateStat = false;
                if isfield(panels,'wins1')
                    angles = get(panels.wins1, 'View');
                    angles(1) = mod((floor(angles(1) / 30) + 1) * 30, 360);
                    set(panels.wins1, 'View', angles);
                    if isfield(panels,'wins2'); set(panels.wins2, 'View', angles); end
                    isRedraw = true;
                end
            case 'S+l'
                isRotateStat = ~isRotateStat;

            case 'hyphen'
                SIZE = max(round(SIZE / 2), 32);
                MID = floor(SIZE / 2);
                RAND_BORDER = floor(SIZE / 6);

                world = zeros(SIZE, SIZE);
                potential = zeros(SIZE, SIZE);
                delta = -ones(SIZE, SIZE);

                [name, R, peaks, mu, sigma, dt, deltaType, kernelType, cells] = LoadCells('0');
                if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                world = AddCells(world, cells);

                [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);
                [fig, panels] = SetUIMode(uiMode, mainPanel, fig, panels, SIZE);
                if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                isNewWorld = true;

            case 'equal'
                SIZE = min(round(SIZE * 2), 4096);
                MID = floor(SIZE / 2);
                RAND_BORDER = floor(SIZE / 6);

                world = zeros(SIZE, SIZE);
                potential = zeros(SIZE, SIZE);
                delta = -ones(SIZE, SIZE);

                [name, R, peaks, mu, sigma, dt, deltaType, kernelType, cells] = LoadCells('0');
                if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                world = AddCells(world, cells);

                [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType);
                [fig, panels] = SetUIMode(uiMode, mainPanel, fig, panels, SIZE);
                if isfield(panels,'winp3'); set(panels.winp3, 'CLim',[0 mu*1.5]); end
                isNewWorld = true;
                
            case ''
            otherwise
                disp(['key[' key ']'])
        end

        if isLastRandom
            switch key
                case {'q','a','w','s','S+q','S+a','S+w','S+s'}
                    world = zeros(SIZE, SIZE);
                    world = RandomPatches(world, R, RAND_BORDER);
                    isNewWorld = true;
            end
        end

        if ~strcmp(key, '') || isNewWorld || isNewSec
            if isCalc; checked = 'on'; else; checked = 'off'; end
            set(menus.('StartCalc'), 'Checked',checked);
            if isCalcStat; checked = 'on'; else; checked = 'off'; end
            set(menus.('StartStat'), 'Checked',checked);
            if isRotateStat; checked = 'on'; else; checked = 'off'; end
            set(menus.('RotateStat'), 'Checked',checked);
            if isAutoCenter; checked = 'on'; else; checked = 'off'; end
            set(menus.('AutoCenter'), 'Checked',checked);
        end
        
        peakSt = regexprep(strtrim(rats(peaks)), ' +', ',');
        if isfield(panels,'wina1') && (~strcmp(key, '') || isNewWorld || isNewSec)
            kernelFuncSt = '$$\left(4r(1-r)\right)^4$$';
            deltaFuncSt = '$$2\left(1 - {|n-\mu|^2 \over 9\sigma^2}\right)^4-1$$';
            st = {
                '\bf Parameters:\rm';
                ['$\begin{tabular}{rrl}' ...
                'World size& &' sprintf('%d', SIZE) '$\times$' sprintf('%d', SIZE) '\\'  ...
                'Space resolution&$R$ =&' sprintf('%d', R) ' cells in 1 mm\\' ...
                'Time resolution&$T$ =&' sprintf('%d', round(1/dt)) ' steps in 1 s\\' ...
                'Time step&$dt$ =&' sprintf('%g', dt) ' s\\' ...
                'Kernel peaks&$\beta$ =&\{' peakSt '\}\\' ...
                'Growth center&$\mu$ =&' sprintf('%g', mu) '\\' ...
                'Growth width&$\sigma$ =&' sprintf('%g', sigma) '\\' ...
                '\end{tabular}$'];
                '';
                '\bf Formulae:\rm';
                ['$\begin{tabular}{rrl}' ...
                'Transition rule&$${\partial \mathbf{f} \over \partial t} (\vec{x}) =$$&$$\delta\left(\mathbf{n} \ast \mathbf{f}(\vec{x})\right)$$ \\' ...
                'Delta function&$$\delta(n) =$$&' deltaFuncSt ' \\' ...
                'Kernel function&$$\mathbf{n}(\vec{u}) =$$&$${\kappa(||\vec{u}||_2) \over \int \kappa}$$ \\' ...
                'Kernel shell function&$$\kappa(r) =$$&$$\beta_{\lfloor br \rfloor} \cdot \gamma ( br \bmod 1 )$$ \\' ...
                'Kernel core function&$$\gamma(r) =$$&' kernelFuncSt ' \\' ...
                '\end{tabular}$'];
                '';
                '\bf Life form:\rm';
                ['  \it ' name '\rm'];
                };
            panels.wina1.String = st;
        end
        key = '';

        tick = tick + 1;
        if isRotateStat && mod(tick, 10) == 0
            if isfield(panels,'wins1')
                angles = get(panels.wins1, 'View');
                angles(1) = mod(angles(1) + 0.5, 360);
                set(panels.wins1, 'View', angles);
                if isfield(panels,'wins2'); set(panels.wins2, 'View', angles); end
            end
        end

        if isCalc || isCalcOnce || isNewSec || isNewWorld || isRedraw
            worldFFT = fft2(world);
            %potential = circshift(real(ifft2(kernelFFT .* worldFFT)), [MID+1, MID+1]);
            prodFFT = kernelFFT .* worldFFT;
            iFFT = ifft2(prodFFT);
            potential = circshift(real(iFFT), [MID+1, MID+1]);
            %potential = circshift(conv2(world, kernel, 'same') / kernelSum, [1,1]);
            delta = DeltaFunc(potential, mu, sigma, deltaType);
            if ~isNewWorld && ~isNewSec && ~isRedraw
                world = max(0, min(1, world + delta * dt));
                gen = gen + 1;
                if isCalcStat
                    [m, m00, g00, mD, gD, mDA, gDA, mo1] = CalcStats(world, delta, 0, false);
                    calcNum = calcNum + 1;
                    if gen > 100 && calcNum > 5
                        statVal = [seq, mu, sigma, m00/R^2, g00/R^2, mD/dt/R, gD/R, mDA/dt, gDA/dt, mo1];  %, real(m), imag(m)
                        statIdx = statIdx + 1;
                        stat(statIdx, :) = statVal;
                        stftVal = [m00/R^2, g00/R^2, mD/dt/R, gD/R, mo1];
                        stftIdx = stftIdx + 1;
                        stft(stftIdx, :) = stftVal;
                    end
                else
                    [m, m00] = CalcStats(world, [], 0, false);
                    calcNum = 0;
                end
                if isAutoCenter
                    world = CenterWorld(world, m);
                end
                if isLastRandom
                    if m00 == 0
                        key = 'w';
                    elseif sum(sum(world(1:20,1:20))) * sum(sum(world(MID:MID+20,MID:MID+20))) > 0
                        key = 's';
                    end
                end
            elseif isNewWorld
                gen = 0;
                calcNum = 0;
                statIdx = 1;
                lastIdx = 1;
                stat = NaN(1, statNum);
                stftIdx = 0;
                stft = [];
                [m, m00] = CalcStats(world, [], 0, true);
                TransformWorld(world, R, 0, 0, 0, 0, 0, false);
            elseif isNewSec
                gen = 0;
                calcNum = 0;
                statIdx = statIdx + 1;
                stat(statIdx, :) = NaN(1, statNum);
                lastIdx = statIdx;
                seq = seq + 1;
                stftIdx = 0;
                stft = [];
                [m, m00] = CalcStats(world, [], 0, true);
            elseif isRedraw
                [m, m00] = CalcStats(world, [], 0, false);
            end

            if isfield(panels,'fig1'); set(panels.fig1, 'CData', world); end
            if isfield(panels,'fig2'); set(panels.fig2, 'CData', delta); end
            if isfield(panels,'fig3'); set(panels.fig3, 'CData', potential); end
            
            if isNewWorld || isNewSec || isRedraw
                if isfield(panels,'fig4'); set(panels.fig4, 'CData', kernel); end
            end
            if mod(statIdx-lastIdx, 10) == 1 || isCalcOnce || isNewSec || isNewWorld || isRedraw
                if isfield(panels,'figs1')
                    set(panels.figs1, 'XData',stat(:,4), 'YData',stat(:,5), ...
                        'ZData',stat(:,10), 'CData',stat(:,10));
                end
                if isfield(panels,'figs2')
                    set(panels.figs2, 'XData',stat(:,8), 'YData',stat(:,7), ...
                        'ZData',stat(:,6), 'CData',stat(:,6));
                end
            end
            if isfield(panels,'figs3') && isCalcStat && stftIdx > 0 && mod(stftIdx, 128) == 0
                [fftX, fftY, fftMaxX, fftMaxY, fftN] = CalcPeriodogram(stft, dt);
                for k = 1:stftNum
                    set(panels.figs3(k), 'XData', fftX, 'YData', fftY(:,k));
                end
                panels.figs3Max.String = ['\leftarrow Period \approx ' sprintf('%.4f', 1/fftMaxX) 's'];
                panels.figs3Max.Position = [fftMaxX fftMaxY 0];
                %wins3XLabel.String = sprintf('freq (from last %d steps)', fftN);
                %periodSt = ['Period = ' sprintf('%.4f', 1/fftMaxX) ' (last ' sprintf('%d', fftN) ' steps)'];
            end
            
            if isfield(panels,'wina2')
                ruleSt = sprintf('R=%d %c={%s} %c=%g %c=%g dt=%g %c=%s %c=%s',...
                    R, char(946), peakSt, char(956), mu, char(963), sigma, dt, char(954), KERNEL_IDS{kernelType}, char(948), DELTA_IDS{deltaType});
                infoSt = sprintf('t=%.2f s, m=%.4f mg', ...
                    gen * dt, m00/R^2);
                if uiMode == 1; modeSt = 'Simple mode';
                elseif uiMode == 2; modeSt = 'Normal mode';
                elseif uiMode == 3; modeSt = 'Advanced mode';
                end
                panels.wina2.String = ['Parameters: ' ruleSt '  |  Life form: ' name newline() modeSt '  |  Status: ' infoSt];
            end
            
            %time 4.53
%            set(panels.winp1.Title, 'String', ['World f(x) (t=' sprintf('%.2f', gen * dt) ' s)']);
        end

        isCalcOnce = false;
        isNewWorld = false;
        isNewSec = false;
        isRedraw = false;

        %drawnow limitrate nocallbacks;
        if SIZE <= 256; pause(0.001); else; drawnow; end
    end
%end

function [fig, panels] = SetUIMode(uiMode, mainPanel, fig, panels, SIZE)
    for p = fieldnames(panels)'
        delete(panels.(p{1}));
    end
    panels = struct;

	if uiMode <= 2
        set(fig, 'Position',[0 0 650 700]);
    elseif uiMode == 3
        set(fig, 'Position',[0 0 1400 700]);
    end

    panels.wina2 = uicontrol('Style','text', 'String','', 'HorizontalAlignment','left', 'FontSize',12);
    init = zeros(SIZE, SIZE);
    init1 = -ones(SIZE, SIZE);
    if uiMode >= 2 || (uiMode == 1 && mainPanel == 1)
        panels.winp1 = subplot(2,4,1); panels.fig1 = imagesc(init);
        title('World f(x)'); caxis([0 1]); axis square; axis off
    end
    if uiMode >= 2 || (uiMode == 1 && mainPanel == 2)
        panels.winp2 = subplot(2,4,2); panels.fig2 = imagesc(init1);
        title('Growth \delta(n \ast f(x))'); caxis([-1 1]); axis square; axis off
    end
    if uiMode >= 2 || (uiMode == 1 && mainPanel == 3)
        panels.winp3 = subplot(2,4,5); panels.fig3 = imagesc(init);
        title('Potential n \ast f(x)'); caxis([0 0.3]); axis square; axis off
    end
    if uiMode >= 2
        panels.winp4 = subplot(2,4,6); panels.fig4 = imagesc(init);
        title('Kernel n(u)'); caxis([0 1]); axis square; axis off
    end
    if uiMode >= 3
        panels.wins1 = subplot(2,4,3);
        panels.figs1 = patch(NaN, NaN, NaN, NaN, 'EdgeColor','interp', 'FaceColor','none');
        xlabel('m = Total mass (mg)');
        ylabel('g = Total growth (mg/s)');
        zlabel('I = Moment of inertia');
        grid on; if ~isOctave; camproj perspective; end

        panels.wins2 = subplot(2,4,7);
        panels.figs2 = patch(NaN, NaN, NaN, NaN, 'EdgeColor','interp', 'FaceColor','none');
        xlabel('\omega_m = Angular speed (deg/s)');
        ylabel('d_g = Growth distance (mm)'); 
        zlabel('s_m = Centroid speed (mm/s)');
        grid on; if ~isOctave; camproj perspective; end
        %plot3(x,y,z,'-b'), scatter3(x,y,z,2,c)

        panels.wins3 = subplot(2,4,4);
        panels.figs3 = plot(0, zeros(1,7));
        title('Periodogram');
        xlabel('Frequency (Hz)');
        ylabel('Power Spectral Density (dB/Hz)'); 
        panels.figs3Max = text(0, 0, '');
        ylim([-140 20]); grid on;
        set(gca, 'XMinorGrid','on', 'MinorGridLineStyle','-', 'MinorGridAlpha',0.05, 'GridAlpha',0.2);

        panels.wina1 = annotation('textbox',[0 0 0 0], 'String','', 'Interpreter','latex', 'FontSize',12);
        %plta2 = annotation('textbox',[0 0 0 0], 'String','', 'Interpreter', 'none');
    end

    y2 = 50;
    if uiMode == 1
        if isfield(panels,'winp1'); setpixelposition(panels.winp1, [20,20+y2,600,600]); end
        if isfield(panels,'winp2'); setpixelposition(panels.winp2, [20,20+y2,600,600]); end
        if isfield(panels,'winp3'); setpixelposition(panels.winp3, [20,20+y2,600,600]); end
    else
        if isfield(panels,'winp1'); setpixelposition(panels.winp1, [10,330+y2,300,300]); end
        if isfield(panels,'winp2'); setpixelposition(panels.winp2, [330,330+y2,300,300]); end
        if isfield(panels,'winp3'); setpixelposition(panels.winp3, [10,10+y2,300,300]); end
    end
    if isfield(panels,'winp4'); setpixelposition(panels.winp4, [330,10+y2,300,300]); end
    if isfield(panels,'wins1'); setpixelposition(panels.wins1, [700,370+y2,230,280]); end
    if isfield(panels,'wins2'); setpixelposition(panels.wins2, [700,50+y2,230,280]); end
    if isfield(panels,'wins3'); setpixelposition(panels.wins3, [1030,360+y2,350,270]); end
    if isfield(panels,'wina1'); setpixelposition(panels.wina1, [1000,10+y2,380,310]); end
    if isfield(panels,'wina2'); setpixelposition(panels.wina2, [10,10,600,40]); end
    %setpixelposition(plta2, [1000,10,380,100]);

    if isfield(panels,'wins1'); set(panels.wins1, 'View', [-37.5, 30]); end
    if isfield(panels,'wins2'); set(panels.wins2, 'View', [-37.5, 30]); end
end

function menus = AddCustomMenu(fig)
    S = char(8679);
    BETA = char(946);
    MU = char(956);
    SIGMA = char(963);
    %set(fig, 'ToolBar','none', 'MenuBar','none');
    if ~isOctave
        menu1 = uimenu(fig, 'Text',[char(10026) ' Lenia']);
        menus.('StartCalc') = uimenu(menu1, 'Text','Start/Stop [enter]', 'MenuSelectedFcn','key=''return'';');
        uimenu(menu1, 'Text','Once [space]', 'MenuSelectedFcn','key=''space'';');
        uimenu(menu1, 'Text','Quit [esc]', 'MenuSelectedFcn','key=''escape'';');
        %------
        uimenu(menu1, 'Text','Inc World Size [=]', 'MenuSelectedFcn','key=''equal'';', 'Separator','on');
        uimenu(menu1, 'Text','Dec World Size [-]', 'MenuSelectedFcn','key=''hyphen'';');
        %------
        menus.('StartStat') = uimenu(menu1, 'Text','Advanced Mode / Start/Stop Stats [J]', 'MenuSelectedFcn','key=''j'';', 'Separator','on');
        uimenu(menu1, 'Text',['Simple Mode / Change Panel [' S 'J]'], 'MenuSelectedFcn','key=''S+j'';');
        uimenu(menu1, 'Text','Clear Last Segment [K]', 'MenuSelectedFcn','key=''k'';', 'Separator','on');
        uimenu(menu1, 'Text',['Clear All [' S 'K]'], 'MenuSelectedFcn','key=''S+k'';');
        uimenu(menu1, 'Text','Rotate 3D Figures [L]', 'MenuSelectedFcn','key=''l'';');
        menus.('RotateStat') = uimenu(menu1, 'Text',['Auto Rotate [' S 'L]'], 'MenuSelectedFcn','key=''S+l'';');

        menu2 = uimenu(fig, 'Text','Load');
        for i = 0:9
            name = LoadCells(num2str(i),0);
            if i == 0; name = [name ' (default)']; end
            uimenu(menu2, 'Text',sprintf('Load %s [%d]',name,i), 'MenuSelectedFcn',sprintf('key=''%d'';',i));
        end
        for i = 0:9
            name = LoadCells(['S+' num2str(i)],0);
            menu = uimenu(menu2, 'Text',sprintf(['Load %s [' S '%d]'],name,i), 'MenuSelectedFcn',sprintf('key=''S+%d'';',i));
            if i==0; set(menu, 'Separator','on'); end
        end

        menu3 = uimenu(fig, 'Text','World');
        menus.('AutoCenter') = uimenu(menu3, 'Text','Auto Center [M]', 'MenuSelectedFcn','key=''m'';');
        uimenu(menu3, 'Text',['Center [' S 'M]'], 'MenuSelectedFcn','key=''S+m'';');
        %------
        uimenu(menu3, 'Text','Random World [N]', 'MenuSelectedFcn','key=''n'';', 'Separator','on');
        uimenu(menu3, 'Text','Clear World [delete]', 'MenuSelectedFcn','key=''backspace'';');
        %------
        uimenu(menu3, 'Text','Load Last [Z]', 'MenuSelectedFcn','key=''z'';', 'Separator','on');
        uimenu(menu3, 'Text','Add Last [X]', 'MenuSelectedFcn','key=''x'';');
        %------
        uimenu(menu3, 'Text','Copy World [^C]', 'MenuSelectedFcn','key=''C+c'';', 'Separator','on');
        uimenu(menu3, 'Text','Paste World [^V]', 'MenuSelectedFcn','key=''C+v'';');
        uimenu(menu3, 'Text','Save World (cells.csv) [^S]', 'MenuSelectedFcn','key=''C+s'';');
        uimenu(menu3, 'Text','Load World (cells.csv) [^L]', 'MenuSelectedFcn','key=''C+l'';');
        %------
        uimenu(menu3, 'Text','Move Up [up]', 'MenuSelectedFcn','key=''uparrow'';', 'Separator','on');
        uimenu(menu3, 'Text',['    ...slightly [' S 'up]'], 'MenuSelectedFcn','key=''S+uparrow'';');
        uimenu(menu3, 'Text','Move Down [down]', 'MenuSelectedFcn','key=''downarrow'';');
        uimenu(menu3, 'Text',['    ...slightly [' S 'down]'], 'MenuSelectedFcn','key=''S+downarrow'';');
        uimenu(menu3, 'Text','Move Left [left]', 'MenuSelectedFcn','key=''leftarrow'';');
        uimenu(menu3, 'Text',['    ...slightly [' S 'left]'], 'MenuSelectedFcn','key=''S+leftarrow'';');
        uimenu(menu3, 'Text','Move Right [right]', 'MenuSelectedFcn','key=''rightarrow'';');
        uimenu(menu3, 'Text',['    ...slightly [' S 'right]'], 'MenuSelectedFcn','key=''S+rightarrow'';');
        %------
        uimenu(menu3, 'Text','Rotate Anti-clockwise [home]', 'MenuSelectedFcn','key=''home'';', 'Separator','on');
        uimenu(menu3, 'Text',['    ...slightly [' S 'home]'], 'MenuSelectedFcn','key=''S+home'';');
        uimenu(menu3, 'Text','Rotate Clockwise [end]', 'MenuSelectedFcn','key=''end'';');
        uimenu(menu3, 'Text',['    ...slightly [' S 'end]'], 'MenuSelectedFcn','key=''S+end'';');
        %------
        uimenu(menu3, 'Text','Flip Horizontally [pgup]', 'MenuSelectedFcn','key=''pageup'';', 'Separator','on');
        uimenu(menu3, 'Text','Flip Vertically [pgdn]', 'MenuSelectedFcn','key=''pagedown'';');
        uimenu(menu3, 'Text',['Mirror Horizontally [' S 'pgup]'], 'MenuSelectedFcn','key=''S+pageup'';');
        uimenu(menu3, 'Text',['Flip Half Vertically [' S 'pgdn]'], 'MenuSelectedFcn','key=''S+pagedown'';');

        menu4 = uimenu(fig, 'Text','Params');
        uimenu(menu4, 'Text','Zoom in (R+) [R]', 'MenuSelectedFcn','key=''r'';');
        uimenu(menu4, 'Text',['    ...slightly [' S 'R]'], 'MenuSelectedFcn','key=''S+r'';');
        uimenu(menu4, 'Text','Zoom out (R-) [F]', 'MenuSelectedFcn','key=''f'';');
        uimenu(menu4, 'Text',['    ...slightly [' S 'F]'], 'MenuSelectedFcn','key=''S+f'';');
        %------
        uimenu(menu4, 'Text','Slow Time (dt--) [T]', 'MenuSelectedFcn','key=''t'';', 'Separator','on');
        uimenu(menu4, 'Text',['Slower Time (dt-) [' S 'T]'], 'MenuSelectedFcn','key=''S+t'';');
        uimenu(menu4, 'Text','Fast Time (dt+) [G]', 'MenuSelectedFcn','key=''g'';');
        uimenu(menu4, 'Text',['Faster Time (dt++) [' S 'G]'], 'MenuSelectedFcn','key=''S+g'';');
        %------
        uimenu(menu4, 'Text',['Inc Growth Center (' MU '+) [Q]'], 'MenuSelectedFcn','key=''q'';', 'Separator','on');
        uimenu(menu4, 'Text',['    ...slightly [' S 'Q]'], 'MenuSelectedFcn','key=''S+q'';');
        uimenu(menu4, 'Text',['Dec Growth Center (' MU '-) [A]'], 'MenuSelectedFcn','key=''a'';');
        uimenu(menu4, 'Text',['    ...slightly [' S 'A]'], 'MenuSelectedFcn','key=''S+a'';');
        %------
        uimenu(menu4, 'Text',['Inc Growth Width (' SIGMA '+) [W]'], 'MenuSelectedFcn','key=''w'';', 'Separator','on');
        uimenu(menu4, 'Text',['    ...slightly [' S 'W]'], 'MenuSelectedFcn','key=''S+w'';');
        uimenu(menu4, 'Text',['Dec Growth Width (' SIGMA '-) [S]'], 'MenuSelectedFcn','key=''s'';');
        uimenu(menu4, 'Text',['    ...slightly [' S 'S]'], 'MenuSelectedFcn','key=''S+s'';');
        %------
        uimenu(menu4, 'Text','Fewer Peaks ([)', 'MenuSelectedFcn','key=''leftbracket'';', 'Separator','on');
        uimenu(menu4, 'Text','More Peaks (])', 'MenuSelectedFcn','key=''rightbracket'';');
        uimenu(menu4, 'Text',['Inc Peak 1 (' BETA '1+) [Y]'], 'MenuSelectedFcn','key=''y'';');
        uimenu(menu4, 'Text',['Dec Peak 1 (' BETA '1-) [' S 'Y]'], 'MenuSelectedFcn','key=''S+y'';');
        uimenu(menu4, 'Text',['Inc Peak 2 (' BETA '2+) [U]'], 'MenuSelectedFcn','key=''u'';');
        uimenu(menu4, 'Text',['Dec Peak 2 (' BETA '2-) [' S 'U]'], 'MenuSelectedFcn','key=''S+u'';');
        uimenu(menu4, 'Text',['Inc Peak 3 (' BETA '3+) [I]'], 'MenuSelectedFcn','key=''i'';');
        uimenu(menu4, 'Text',['Dec Peak 3 (' BETA '3-) [' S 'I]'], 'MenuSelectedFcn','key=''S+i'';');
        uimenu(menu4, 'Text',['Inc Peak 4 (' BETA '4+) [O]'], 'MenuSelectedFcn','key=''o'';');
        uimenu(menu4, 'Text',['Dec Peak 4 (' BETA '4-) [' S 'O]'], 'MenuSelectedFcn','key=''S+o'';');
        uimenu(menu4, 'Text',['Inc Peak 5 (' BETA '5+) [P]'], 'MenuSelectedFcn','key=''p'';');
        uimenu(menu4, 'Text',['Dec Peak 5 (' BETA '5-) [' S 'P]'], 'MenuSelectedFcn','key=''S+p'';');
        %------
        uimenu(menu4, 'Text','Random Parameters [B]', 'MenuSelectedFcn','key=''b'';', 'Separator','on');
    end
end

% =========== UTILS ===========

function retval = isOctave
  persistent cacheval;  % speeds up repeated calls

  if isempty(cacheval)
    cacheval = (exist ("OCTAVE_VERSION", "builtin") > 0);
  end

  retval = cacheval;
end

% =========== UI ===========

function KeyPress(~, e)
    global key
    if strcmp(e.Key, '0') && isempty(e.Character)
        key = '';
    else
        key = e.Key;
        if ismember('shift', e.Modifier); key = ['S+' key]; end
        if ismember('control', e.Modifier); key = ['C+' key]; end
        if ismember('alt', e.Modifier); key = ['A+' key]; end
        if ismember('command', e.Modifier); key = ['M+' key]; end
    end
end

% =========== CALCULATION ===========

function gamma = KernelCore(r, kernelType)
    rm = min(r, 1);
    if kernelType==1
        gamma = (4 * rm .* (1-rm)).^4;
    elseif kernelType==2
        gamma = exp( 4 - 1 ./ (rm .* (1-rm)) );
    end
end

function kappa = KernelShell(r, peaks, kernelType)
    k = size(peaks,2);
    kr = k * r;
    peak = peaks(min(floor(kr) + 1,k));
    kappa = (r<1) .* KernelCore(mod(kr, 1), kernelType) .* peak;
end

function delta = DeltaFunc(n, mu, sigma, deltaType)
    if deltaType==1
        %time 0.86
        delta = max(0, 1 - (n - mu).^2 ./ (sigma^2 * 9) ).^4 * 2 - 1;
    elseif deltaType==2
        delta = exp( - (n - mu).^2 ./ (sigma^2 * 2) ) * 2 - 1;
    end
end

function [kernel, kernelFFT] = CalcKernel(SIZE, R, peaks, kernelType)
    MID = floor(SIZE / 2);
    J = repmat((1:SIZE), SIZE, 1);
    X = (J-MID)/R;
    Y = X';
    D = sqrt(X.^2 + Y.^2);

    kernel = KernelShell(D, peaks, kernelType);
    kernelSum = sum(sum(kernel));
    kernelNorm = kernel / kernelSum;
    kernelFFT = fft2(kernelNorm);
end

% =========== MOMENTS ===========

function [m, m00, g00, mD, gD, mDA, gDA, mo1] = CalcStats(world, delta, shift, isReset)
    SIZE = size(world, 1);
    MID = floor(SIZE / 2);

    persistent om omA ogA
    if isempty(om) || isReset
        om = complex(MID, MID);
        omA = 0;
        ogA = 0;
    end
    om = om + shift;
    
    J = repmat((1:SIZE), SIZE, 1);
    I = J';
    mx = floor(real(om));
    my = floor(imag(om));
    %time 0.83
    X = J + (J<mx-MID)*SIZE - (J>mx+MID)*SIZE;
    %time 0.70
    Y = I + (I<my-MID)*SIZE - (I>my+MID)*SIZE;

    m00 = sum(sum(world));
    x10 = world .* X; m10 = sum(sum(x10));
    x01 = world .* Y; m01 = sum(sum(x01));
    if m00 > 0
        m = complex(m10, m01) / m00;
    else
        m = complex(MID, MID);
    end
    
    if ~isempty(delta)
        x20 = x10 .* X; m20 = sum(sum(x20));
        x02 = x01 .* Y; m02 = sum(sum(x02));

        growth = max(0, delta);
        g00 = sum(sum(growth));
        g10 = sum(sum(growth .* X));
        g01 = sum(sum(growth .* Y));
        if g00 > 0
            g = complex(g10, g01) / g00;
        else
            g = complex(MID, MID);
        end

        dm = m - om;
        dg = m - g;
        mD = abs(dm);
        gD = abs(dg);
        mA = angle(dm) / pi * 180;
        gA = angle(dg) / pi * 180;
        mDA = mA - omA;  mDA = mod(mDA + 540, 360) - 180;
        gDA = gA - ogA;  gDA = mod(gDA + 540, 360) - 180;

        mu20 = m20 - real(m)*m10;
        mu02 = m02 - imag(m)*m01;
        mo1 = (mu20 + mu02) / m00^2;

        omA = mA;
        ogA = gA;
    end
    m = complex(mod(real(m), SIZE), mod(imag(m), SIZE));
    om = m;
end

function [fftX, fftY, fftMaxX, fftMaxY, fftN] = CalcPeriodogram(stft, dt)
    %persistent maN mapsd
    Fs = 1/dt;
    fftN = size(stft,1);
    %fftN = min(128, fftN);
    %fftN = pow2(floor(log2(fftN)));
    fftN2 = floor(fftN/2);
    fftX = (0:fftN2)*Fs/fftN;

    %fftY = fft(stft(end-fftN+1:end,:));
    %fftY2 = [NaN(1,stftNum); 10*log10( abs(fftY(2:fftN2+1,:)).^2 /Fs/fftN*2 )];
    dft = fft(stft(end-fftN+1:end,:));
    dft = dft(1:fftN2+1,:);
    psd = abs(dft).^2 /Fs/fftN;
    psd(2:end-1,:) = 2*psd(2:end-1,:);

    %{
    if isempty(maN)
        mapsd = psd;
        maN = 1;
    else
        mapsd = ((mapsd * maN) + psd) / (maN+1);
        maN = maN + 1;
    end
    %}
    
    fftY = 10*log10(psd);
    fftY(1,:) = NaN;

    [fftMax, fftMaxI] = max(fftY);
    [fftMax2, fftMax2I] = max(fftMax);
    fftMaxX = fftX(fftMaxI(fftMax2I));
    fftMaxY = fftMax2;
end

% =========== WORLD ===========

function world = RandomWorld(world, border, max)
    SIZE = size(world, 1);
    range = border:(SIZE-border);
    world = zeros(SIZE, SIZE);
    rands = rand(SIZE, SIZE) * max;
    world(range, range) = rands(range, range);
end

function world = RandomPatches(world, R, border)
    randSize = floor(R * 0.9);
    SIZE = size(world, 1);
    range = [border SIZE-border-randSize];
    world = zeros(SIZE, SIZE);
    for k = 1:30
        rands = rand(randSize, randSize) * (rand()*0.5+0.5);
        r = randi(range);
        c = randi(range);
        world((1:randSize)+r, (1:randSize)+c) = rands(1:randSize, 1:randSize);
    end
end

function [cells2, R] = DoubleCells(cells, R)
    w = size(cells, 2) * 2;
    h = size(cells, 1) * 2;
    cells2 = zeros(h, w);
    cells2(1:2:h, 1:2:w) = cells;
    cells2(1:2:h, 2:2:w) = cells;
    cells2(2:2:h, 1:2:w) = cells;
    cells2(2:2:h, 2:2:w) = cells;
    R = R * 2;
end

function [cells2, R] = HalfCells(cells, R)
    w = size(cells, 2);
    h = size(cells, 1);
    cells2 = cells(1:2:h, 1:2:w);
    R = ceil(R / 2);
end

function world = TranslateWorld(world, dX, dY)
    world = circshift(world, [floor(dY), floor(dX)]);
end

function [world, R] = TransformWorld(world, R, dX, dY, dS, dA, flip, isAccu)
	%flip:1(horiz), 2(vert), 3(h+v), 4(mirror horiz), 5(mirror+flip horiz), 6(mirror diag)
    persistent txS txA oldWorld lastAccu
    SIZE = size(world, 1);
    MID = floor(SIZE / 2);

    if dS == 0
        lastAccu = false;
        return
    end
    
    if dS == 1 && dA == 0 && flip == 0
        world = TranslateWorld(world, dX, dY);
        return
    end
    
    R2 = floor(R * dS);
    if R2 < 2 || R2 > MID
        return
    end
    R = R2;

    if isempty(txS) || ~(isAccu && lastAccu)
        txS = 1;
        txA = 0;
        oldWorld = zeros(SIZE, SIZE);
        oldWorld(:,:) = world(:,:);
    end
    lastAccu = isAccu;
    txS = txS * dS;
    txA = txA + dA;

    world = zeros(SIZE, SIZE);

    if txS >= 1
        j = 0:SIZE-1;
    else
        j = floor(MID * (1-txS)):floor(MID * (1+txS))-1;
    end
    J = repmat(j, length(j), 1);
    Z1 = complex(J', J);
    zX = complex(dX, dY);
    zA = complex(cosd(txA), sind(txA)); 
    zM = complex(MID, MID);
    Z2 = floor((Z1 - zM) / txS / zA - zX);

    J2 = real(Z2);
    I2 = imag(Z2);
    switch flip
        case 1; J2 = -J2;
        case 2; I2 = -I2;
        case 3; J2 = -J2; I2 = -I2;
        case 4; J2 = abs(J2);
        case 5; I2 = I2 .* ((J2>0)*2-1);
    end
    J2 = J2 + MID;
    I2 = I2 + MID;
    
    idx1 = sub2ind(size(world), mod(imag(Z1),SIZE)+1, mod(real(Z1),SIZE)+1);
    idx2 = sub2ind(size(world), mod(I2,SIZE)+1, mod(J2,SIZE)+1);
    world(idx1(:)) = oldWorld(idx2(:));
end

function world = CenterWorld(world, m)
    SIZE = size(world, 1);
    MID = floor(SIZE / 2);
    d = complex(MID, MID) - m;
	world = TranslateWorld(world, real(d), imag(d));
    CalcStats(world, [], d, false);
end

function world = AddCells(world, cells)
    SIZE = size(world, 1);
    MID = floor(SIZE / 2);
    w = size(cells, 2);
    h = size(cells, 1);
    i = MID - floor(w / 2);
    j = MID - floor(h / 2);
    if w >= SIZE || h >= SIZE
        return
    end

    world(j:j+h-1, i:i+w-1) = cells;
end

% =========== ANIMALS ===========

function [name, R, peaks, mu, sigma, dt, deltaType, kernelType, cells] = LoadCells(key, isLoadCell)
    deltaType = 1;
    kernelType = 1;
    name = '';
    if ~exist('isLoadCell','var')
        isLoadCell = true;
    end
    switch key
        case '1'; name='Paraptera cavus pedes'; if isLoadCell; R=13;peaks=[1];mu=0.3;sigma=0.045;dt=0.1;cells=[0,0,0,0,0,0,0,0.017183,0.081096,0.118635,0.118118,0.082839,0.037035,0.00529,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.025744,0.210784,0.392454,0.496581,0.529889,0.513557,0.465689,0.36821,0.241666,0.121303,0.038886,0.001942,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.142213,0.427627,0.660647,0.752479,0.720634,0.662269,0.637714,0.644533,0.6465,0.576205,0.424827,0.246361,0.116284,0.040665,0.002517,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.239381,0.579935,0.865447,0.948113,0.83567,0.656797,0.531284,0.502572,0.579706,0.698479,0.764211,0.69186,0.502915,0.297899,0.162452,0.076256,0.021294,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.258195,0.620359,0.97324,1,1,0.780956,0.524312,0.389308,0.383696,0.510515,0.722422,0.901644,0.897904,0.714003,0.47561,0.299991,0.188029,0.104639,0.03143,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.165912,0.502151,0.898536,1,1,1,0.727666,0.442951,0.325062,0.359835,0.542805,0.839773,1,1,0.873045,0.620827,0.429942,0.311442,0.221455,0.133239,0.041413,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0.001671,0.22808,0.593886,1,1,1,1,0.719795,0.44614,0.369472,0.452171,0.685886,1,1,1,1,0.767685,0.587173,0.464066,0.366608,0.270103,0.168128,0.061942,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.139687,0.558947,0.979841,1,1,1,0.746926,0.519054,0.508509,0.640454,0.834735,0.995088,1,1,1,0.986243,0.81488,0.66735,0.534352,0.423726,0.316373,0.202499,0.08538,0.008075,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.035401,0.838209,0.951135,1,1,1,0.662677,0.502462,0.501798,0.580888,0.73583,0.930189,1,1,1,1,1,0.841767,0.658396,0.531714,0.449921,0.354901,0.226569,0.097199,0.01614,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.613488,0.841068,0.929563,1,1,0.997178,0.499388,0.376071,0.399718,0.479812,0.616406,0.799002,0.96553,1,1,1,1,0.793763,0.632857,0.55916,0.532994,0.479943,0.372315,0.226285,0.093923,0.016464,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0.131201,0.616908,0.801933,0.917747,1,1,0.825859,0.314864,0.237522,0.279,0.360965,0.472579,0.610981,0.751224,0.843815,0.86856,0.716245,0.57086,0.491277,0.468883,0.512402,0.59072,0.610312,0.51193,0.347745,0.20051,0.090189,0.018267,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.489208,0.763838,0.902557,0.997424,1,0.591736,0.123711,0.079113,0.143816,0.22537,0.311224,0.381434,0.425077,0.424426,0.16514,0.033733,0.026282,0.120483,0.264828,0.46105,0.671862,0.759718,0.648587,0.468237,0.324825,0.21701,0.121487,0.040872,0,0,0,0,0,0,0,0,0,0,0,0;0,0.378766,0.712004,0.874992,0.981183,0.695283,0.346039,0,0,0.004306,0.079749,0.126137,0.138337,0.098984,0,0,0,0,0,0.081127,0.459338,0.821891,0.935044,0.778194,0.600075,0.494435,0.418794,0.323771,0.209892,0.086166,0.008393,0,0,0,0,0,0,0,0,0,0;0,0.277574,0.612892,0.592964,0.518749,0.390657,0.227662,0.050372,0,0,0,0,0,0,0,0,0,0,0,0,0.548831,1,1,0.907726,0.769603,0.72324,0.67159,0.570042,0.444894,0.29345,0.143361,0.02755,0,0,0,0,0,0,0,0,0;0,0.162004,0.124869,0.330549,0.377886,0.363624,0.303978,0.189584,0.059598,0,0,0,0,0,0,0,0,0,0,0,0.788626,1,1,1,0.989107,0.948694,0.836328,0.69147,0.576312,0.472707,0.346918,0.186655,0.059279,0,0,0,0,0,0,0,0;0,0,0,0.093725,0.214317,0.280569,0.309088,0.2938,0.23751,0.149966,0.059313,0,0,0,0,0,0,0,0,0.194448,0.994581,1,1,1,1,0.987302,0.745823,0.578222,0.509253,0.505079,0.481578,0.382623,0.234987,0.09893,0.021168,0,0,0,0,0,0;0,0,0,0,0.018575,0.122845,0.203471,0.253799,0.284347,0.290158,0.253819,0.158477,0.042185,0,0,0,0,0,0,0.828115,1,1,1,1,1,0.719552,0.457615,0.340967,0.341877,0.430843,0.519139,0.530688,0.434736,0.288178,0.157722,0.058444,0.000604,0,0,0,0;0,0,0,0,0,0,0.007844,0.080017,0.135641,0.184131,0.233813,0.263126,0.231018,0.119228,0.030092,0,0,0.023045,0.575523,1,1,1,1,0.998391,0.697578,0.342166,0.163659,0.126277,0.197864,0.350611,0.531002,0.643977,0.618263,0.498036,0.356424,0.218803,0.098527,0.016451,0,0,0;0,0,0,0,0,0,0,0,0,0,0.038713,0.136543,0.234407,0.2816,0.264388,0.242946,0.284655,0.614418,1,1,1,1,1,0.680534,0.254023,0.002697,0,0,0.103867,0.321548,0.584987,0.777915,0.800778,0.70268,0.562714,0.410954,0.265704,0.11923,0.016996,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.069949,0.198147,0.281896,0.345117,0.536286,0.960161,0.940196,0.967557,1,1,0.808233,0.184247,0,0,0,0,0.077482,0.381805,0.740059,0.974533,0.987539,0.883939,0.750279,0.590051,0.429878,0.270261,0.108337,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.074793,0.216103,0.733514,0.701507,0.732494,0.861358,0.998345,1,0.275363,0,0,0,0,0,0.172871,0.577091,0.96786,1,1,1,0.892353,0.741969,0.568339,0.398283,0.223195,0.053055,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.458348,0.420131,0.434563,0.564516,0.788756,0.986384,0.795851,0,0,0,0,0,0,0.402466,0.763809,0.977546,1,1,1,1,0.857555,0.672119,0.496128,0.326817,0.136271,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.185788,0.147956,0.135145,0.225751,0.463535,0.757204,0.976999,0.263923,0.003988,0,0,0,0,0.158696,0.513349,0.785309,0.972422,1,1,1,1,0.953715,0.755145,0.570045,0.403499,0.214066,0.034849;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.090079,0.41563,0.733758,0.323216,0.287213,0.051349,0,0,0,0,0.229016,0.521984,0.761074,0.939582,1,1,1,1,1,0.828554,0.631425,0.461845,0.275411,0.073848;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.018378,0.385093,0.146457,0.251194,0.296231,0.154,0.000504,0,0,0,0.220757,0.487057,0.690366,0.848425,0.96285,1,1,1,1,0.918303,0.684269,0.498649,0.307435,0.098889;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.025916,0.15509,0.254051,0.259708,0.137684,0.015769,0,0,0.165104,0.38384,0.547383,0.693913,0.829277,0.944805,1,1,1,1,0.743828,0.516901,0.316557,0.10774;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.021033,0.141244,0.240646,0.274044,0.179856,0.035194,0,0.037601,0.206474,0.333004,0.456151,0.610082,0.800647,0.957726,1,1,1,0.825836,0.512395,0.293694,0.094681;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.103743,0.239528,0.312745,0.197991,0.015186,0,0,0.053015,0.157018,0.316617,0.401414,0.751408,1,1,1,0.929257,0.470484,0.236525,0.063053;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.086353,0.274925,0.335833,0.126387,0,0,0,0,0,0,0.393699,1,1,1,0.991161,0.366913,0.147427,0.020991;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.131405,0.32528,0.284717,0.066247,0,0,0,0,0.005877,0.472653,1,1,1,0.909734,0.188782,0.048453,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.183627,0.322527,0.244269,0.098992,0.032421,0.028754,0.082231,0.247928,0.872739,1,1,1,0.643431,0.014826,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.200976,0.31231,0.310007,0.274953,0.293576,0.402985,0.820389,1,1,0.994944,0.964648,0.245775,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1647,0.291897,0.430128,0.736501,0.994415,0.973002,0.922842,0.863068,0.805454,0.606411,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.31926,0.716342,0.797112,0.77027,0.709913,0.648796,0.601928,0.564209,0.090776,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05954,0.429055,0.460809,0.45957,0.430457,0.397546,0.378053,0.36828,0.235215,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.133562,0.156819,0.157444,0.143153,0.123945,0.127818,0.155197,0.189747,0,0,0,0,0,0]; end
        case '2'; name='Rotorbium'; if isLoadCell; R=13;peaks=[1];mu=0.156;sigma=0.0224;dt=0.1;cells=[0,0,0,0,0,0,0,0,0.003978,0.016492,0.004714,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.045386,0.351517,0.417829,0.367137,0.37766,0.426948,0.431058,0.282864,0.081247,0,0,0,0,0,0;0,0,0,0,0.325473,0.450995,0.121737,0,0,0,0.003113,0.224278,0.47101,0.456459,0.247231,0.071609,0.013126,0,0,0;0,0,0,0.386337,0.454077,0,0,0,0,0,0,0,0.27848,0.524466,0.464281,0.242651,0.096721,0.038476,0,0;0,0,0.258817,0.583802,0.150994,0,0,0,0,0,0,0,0.226639,0.548329,0.550422,0.334764,0.153108,0.087049,0.042872,0;0,0.008021,0.502406,0.524042,0.059531,0,0,0,0,0,0,0.033946,0.378866,0.615467,0.577527,0.357306,0.152872,0.090425,0.058275,0.023345;0,0.179756,0.596317,0.533619,0.162612,0,0,0,0,0.015021,0.107673,0.325125,0.594765,0.682434,0.594688,0.381172,0.152078,0.073544,0.054424,0.030592;0,0.266078,0.614339,0.605474,0.379255,0.195176,0.16516,0.179148,0.204498,0.299535,0.760743,1,1,1,1,0.490799,0.237826,0.069989,0.043549,0.022165;0,0.333031,0.64057,0.686886,0.60698,0.509866,0.450525,0.389552,0.434978,0.859115,0.94097,1,1,1,1,1,0.747866,0.118317,0.037712,0.006271;0,0.417887,0.6856,0.805342,0.824229,0.771553,0.69251,0.614328,0.651704,0.843665,0.910114,1,1,0.81765,0.703404,0.858469,1,0.613961,0.035691,0;0.04674,0.526827,0.787644,0.895984,0.734214,0.661746,0.670024,0.646184,0.69904,0.723163,0.682438,0.618645,0.589858,0.374017,0.30658,0.404027,0.746403,0.852551,0.031459,0;0.130727,0.658494,0.899652,0.508352,0.065875,0.009245,0.232702,0.419661,0.461988,0.470213,0.390198,0.007773,0,0.010182,0.080666,0.17231,0.44588,0.819878,0.034815,0;0.198532,0.810417,0.63725,0.031385,0,0,0,0,0.315842,0.319248,0.321024,0,0,0,0,0.021482,0.27315,0.747039,0,0;0.217619,0.968727,0.104843,0,0,0,0,0,0.152033,0.158413,0.114036,0,0,0,0,0,0.224751,0.647423,0,0;0.138866,1,0.093672,0,0,0,0,0,0.000052,0.015966,0,0,0,0,0,0,0.281471,0.455713,0,0;0,1,0.145606,0.005319,0,0,0,0,0,0,0,0,0,0,0,0.016878,0.381439,0.173336,0,0;0,0.97421,0.262735,0.096478,0,0,0,0,0,0,0,0,0,0,0.013827,0.217967,0.287352,0,0,0;0,0.593133,0.2981,0.251901,0.167326,0.088798,0.041468,0.013086,0.002207,0.009404,0.032743,0.061718,0.102995,0.1595,0.24721,0.233961,0.002389,0,0,0;0,0,0.610166,0.15545,0.200204,0.228209,0.241863,0.243451,0.270572,0.446258,0.376504,0.174319,0.154149,0.12061,0.074709,0,0,0,0,0;0,0,0.354313,0.32245,0,0,0,0.151173,0.479517,0.650744,0.392183,0,0,0,0,0,0,0,0,0;0,0,0,0.329339,0.328926,0.176186,0.198788,0.335721,0.534118,0.549606,0.361315,0,0,0,0,0,0,0,0,0;0,0,0,0,0.090407,0.217992,0.190592,0.174636,0.222482,0.375871,0.265924,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.050256,0.235176,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.180145,0.132616,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.092581,0.188519,0.118256,0,0,0,0]; end
        case '3'; name='Gyrogeminium'; if isLoadCell; R=18;peaks=[1,1,1];mu=0.25;sigma=0.034;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.02191,0.042584,0.056002,0.049798,0.029214,0.015619,0.004298,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01754,0.058081,0.103855,0.143338,0.157897,0.14917,0.120455,0.081754,0.058145,0.046056,0.042396,0.035099,0.023622,0.008674,0,0,0,0.000864,0.005639,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.017653,0.054876,0.099884,0.141947,0.186529,0.21171,0.214307,0.197981,0.171176,0.150873,0.143478,0.145703,0.144855,0.125879,0.081539,0.039643,0.016419,0.010646,0.01414,0.020895,0.027339,0.021355,0.008121,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.043433,0.107783,0.167061,0.210429,0.240096,0.266049,0.282636,0.282967,0.267153,0.245831,0.234624,0.246935,0.282258,0.319942,0.334195,0.306546,0.229246,0.133897,0.075156,0.057567,0.060271,0.062735,0.053377,0.036114,0.02091,0.014587,0.009159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.015782,0.092347,0.184417,0.264462,0.323258,0.359319,0.391907,0.421087,0.443951,0.450041,0.436263,0.414872,0.403924,0.420095,0.46004,0.512243,0.548153,0.546768,0.483642,0.36478,0.264122,0.218327,0.206069,0.186449,0.143721,0.09772,0.069355,0.079344,0.121234,0.155313,0.123584,0.045347,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.010988,0.078606,0.167103,0.256545,0.331268,0.394399,0.443518,0.503222,0.561214,0.616695,0.650312,0.663804,0.655473,0.629705,0.610496,0.61488,0.643656,0.689626,0.732154,0.7255,0.652311,0.562968,0.50288,0.470576,0.435904,0.386359,0.312853,0.254923,0.255658,0.337614,0.441402,0.474993,0.386356,0.196099,0.05026,0.01014,0.008574,0.008357,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.044669,0.110885,0.190532,0.28301,0.375097,0.448848,0.52011,0.589359,0.641369,0.676846,0.696677,0.735971,0.768501,0.757028,0.71888,0.696948,0.714765,0.768399,0.827788,0.840599,0.804641,0.753605,0.730569,0.724849,0.714571,0.69811,0.667579,0.607416,0.564542,0.610939,0.721381,0.789551,0.780478,0.632747,0.33597,0.151731,0.121662,0.139211,0.118104,0.032702,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.029213,0.096469,0.181837,0.297426,0.428847,0.535161,0.606149,0.657159,0.694654,0.706546,0.690109,0.685218,0.744938,0.809038,0.807918,0.762505,0.734147,0.760176,0.827279,0.879313,0.880508,0.845754,0.831086,0.872994,0.933563,0.903117,0.819822,0.831829,0.903605,0.9017,0.888785,0.950242,1,1,1,0.871941,0.484836,0.322607,0.343471,0.356214,0.263609,0.076271,0,0,0,0,0,0,0;0,0,0,0,0,0.023523,0.127085,0.254272,0.414335,0.589537,0.725279,0.783555,0.805529,0.833325,0.838853,0.812209,0.774587,0.781247,0.858916,0.914672,0.890714,0.822955,0.785482,0.82331,0.908848,0.958031,0.943613,0.895086,0.896476,1,1,1,0.755665,0.654914,0.852411,0.995935,1,1,1,0.985218,1,1,1,0.718193,0.614613,0.629355,0.571763,0.358519,0.079044,0,0,0,0,0,0;0,0,0,0,0.006076,0.141583,0.323825,0.510114,0.698257,0.818326,0.807391,0.742837,0.770344,0.886877,0.94445,0.925456,0.896582,0.931176,1,1,1,0.925309,0.839932,0.874184,0.987732,1,1,0.925341,0.935313,1,1,1,0.804286,0.480426,0.487526,0.745317,0.973978,1,0.962296,0.854432,0.85923,0.996833,1,1,0.968921,0.886903,0.843811,0.685012,0.337617,0.057103,0,0,0,0,0;0,0,0,0,0.080057,0.305503,0.513298,0.699863,0.824697,0.822822,0.664177,0.545326,0.623645,0.832298,0.958517,0.964163,0.935755,0.984694,1,1,1,0.993918,0.817138,0.78338,0.926623,1,0.935002,0.731911,0.708136,0.882341,0.997958,1,0.806859,0.323739,0.171089,0.308504,0.67705,0.876628,0.813421,0.646567,0.590693,0.791452,1,1,1,1,1,0.945435,0.650281,0.229735,0.056103,0.008006,0,0,0;0,0,0,0,0.174994,0.432951,0.63734,0.795015,0.894213,0.850081,0.656024,0.538413,0.61707,0.81018,0.909776,0.854054,0.728609,0.722847,0.930676,1,1,0.900933,0.672838,0.558889,0.706216,0.908667,0.724295,0.280961,0.151669,0.339797,0.618444,0.840349,0.689255,0.218991,0,0,0.154616,0.553859,0.606737,0.455162,0.363099,0.508262,0.904918,1,1,1,1,1,0.924688,0.463028,0.166502,0.067178,0.008312,0,0;0,0,0,0.018778,0.261564,0.545766,0.741628,0.888055,0.988787,0.968807,0.803164,0.691722,0.746568,0.872723,0.878083,0.672007,0.423439,0.347734,0.549632,0.947937,0.971248,0.811637,0.543273,0.420353,0.621904,0.952835,0.770288,0,0,0,0,0.255969,0.362691,0.077203,0,0,0,0.065619,0.361202,0.325153,0.256085,0.360569,0.788179,0.987427,0.993433,0.993147,1,1,1,0.719625,0.302115,0.159263,0.055707,0,0;0,0,0,0.083112,0.435263,0.736101,0.889613,1,1,1,0.952401,0.846372,0.86992,0.90658,0.867825,0.584879,0.302811,0.191434,0.367598,0.879638,0.985407,0.841158,0.516453,0.357547,0.59461,0.992366,1,0.181634,0,0,0,0,0,0,0,0,0,0,0.118007,0.255234,0.249996,0.355262,0.768488,0.981932,0.991674,0.989117,1,1,1,0.988568,0.49378,0.270853,0.133722,0.010177,0;0,0,0.014167,0.301076,0.748374,0.932558,0.934159,0.983926,1,0.99529,0.88668,0.741166,0.712205,0.764682,0.74702,0.583091,0.318105,0.196359,0.347789,0.851333,1,0.959889,0.588841,0.276527,0.425458,0.946617,1,0.577749,0,0,0,0,0,0,0,0,0,0,0,0.185012,0.255773,0.370157,0.738405,0.962708,0.986689,0.99014,1,1,1,1,0.737407,0.412916,0.229009,0.042743,0;0,0.014008,0.202806,0.650472,0.99047,0.932946,0.758638,0.725704,0.795428,0.760156,0.542712,0.349314,0.293832,0.439532,0.628731,0.584293,0.383527,0.225346,0.299868,0.749068,1,0.999135,0.638774,0.03428,0.005435,0.433117,0.911905,0.466535,0,0,0,0,0,0,0,0,0,0,0,0.040634,0.158619,0.269697,0.583839,0.880996,0.94607,0.974962,1,1,1,1,0.970191,0.579058,0.337589,0.093011,0;0,0.145903,0.479992,0.89866,1,0.874906,0.612086,0.513707,0.552665,0.512695,0.301546,0.104747,0.052678,0.253499,0.539204,0.540212,0.442597,0.193091,0.119485,0.394402,0.88099,0.965013,0.521221,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.103434,0.387006,0.742164,0.881529,0.948466,1,1,1,1,1,0.735874,0.447891,0.148677,0;0.033452,0.314387,0.6832,0.980914,1,0.92465,0.63485,0.477714,0.479057,0.482589,0.331746,0.154702,0.097736,0.265734,0.447369,0.473835,0.382247,0.161897,0,0,0.227942,0.789546,0.420791,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.028619,0.333323,0.702449,0.876688,0.956761,1,1,1,1,1,0.84544,0.532598,0.193062,0;0.085572,0.444481,0.778323,0.990531,1,1,0.801101,0.53906,0.486538,0.575275,0.518403,0.348099,0.219085,0.256245,0.319362,0.365768,0.307886,0.066511,0,0,0,0.420601,0.870242,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.121785,0.490061,0.837681,0.965006,1,1,1,1,1,1,0.901781,0.571912,0.211889,0.001447;0.116596,0.544332,0.860994,1,1,1,0.867361,0.397234,0.26586,0.489171,0.539567,0.382975,0.190147,0.058683,0.10423,0.19371,0.232937,0.025875,0,0,0,0.105843,0.921521,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.242505,0.677644,0.939517,1,1,1,1,1,1,1,0.913923,0.557479,0.197655,0.003847;0.126909,0.657747,1,1,1,1,0.682431,0,0,0.111616,0.397855,0.24722,0,0,0,0,0.161611,0.147626,0,0,0,0,0.881283,1,0.480819,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.320767,0.773099,0.989903,1,1,1,1,1,1,1,0.891595,0.499533,0.157951,0.002554;0.134032,0.800659,1,1,1,1,0.376806,0,0,0,0.094883,0,0,0,0,0,0.038303,0.33945,0.134464,0,0,0,0,0.882791,0.212618,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.398412,0.85001,1,1,1,1,1,1,1,1,0.845923,0.413997,0.110029,0;0.134511,0.926698,1,1,1,1,0.488835,0,0,0,0,0,0,0,0,0,0,0.283999,0.277186,0,0,0,0,0,0.017702,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.564751,0.944218,1,1,1,1,1,1,1,1,0.775475,0.319454,0.062557,0;0.09326,0.978648,1,1,1,1,0.977756,0.137185,0,0,0,0,0,0,0,0,0,0,0.105559,0.031406,0,0,0,0,0.015886,0.031875,0.037466,0.024366,0,0,0,0,0,0,0,0,0,0,0,0,0.337483,0.94682,1,1,1,1,1,1,1,1,1,0.661427,0.221199,0.021642,0;0,0.896951,1,1,1,1,1,0.647911,0.07799,0,0,0,0,0,0,0,0,0,0,0,0,0.010906,0.072478,0.11482,0.122323,0.149535,0.18304,0.182804,0.11432,0.011324,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0.969052,0.48636,0.12344,0,0;0,0.68185,1,1,1,1,1,0.908649,0.142202,0,0,0.004814,0.010475,0,0,0,0,0,0,0,0.253126,1,0.603834,0.496522,0.364671,0.318951,0.321342,0.332293,0.313584,0.184434,0.043516,0.006365,0.000442,0,0,0,0,0,0.002427,1,1,1,1,1,1,1,1,1,1,1,0.669237,0.273629,0.030577,0,0;0,0.450326,1,1,1,1,1,0.838926,0.125665,0.031654,0.008458,0.025913,0.043958,0.04652,0.0183,0,0,0,0,0,0.161984,1,0.993014,0.700392,0.444369,0.359523,0.320059,0.331722,0.381327,0.375025,0.277495,0.216518,0.217586,0.221686,0.18428,0.080768,0.018064,0.03983,0.335921,1,1,1,1,1,1,1,1,1,1,0.752639,0.342056,0.079062,0,0,0;0,0.361636,0.954327,1,1,0.878514,0.834976,0.690724,0.229189,0.093715,0.033848,0.029578,0.048241,0.087884,0.097587,0.040608,0,0,0,0.031639,0.242968,1,0.942492,0.903567,0.416373,0.337162,0.261994,0.235571,0.286334,0.378884,0.413833,0.423242,0.447008,0.492358,0.526944,0.467247,0.344277,0.408577,0.848632,1,1,1,1,1,1,1,1,0.655025,0.673683,0.335009,0.070896,0,0,0,0;0,0.419497,0.855825,0.969305,1,0.811076,0.7148,0.751883,0.616031,0.248809,0.120255,0.058165,0.063456,0.116384,0.188223,0.182559,0.128647,0.088899,0.08767,0.141582,0.289632,0.438739,0.444499,0.424018,0.375477,0.28893,0.166061,0.083598,0.087474,0.183702,0.295657,0.338304,0.357205,0.412278,0.545191,0.709405,0.760461,0.861204,1,1,1,1,1,1,1,1,0.544721,0.247978,0.224554,0.032244,0,0,0,0,0;0,0.509318,0.758769,0.884139,0.976126,1,0.830854,0.865956,0.97449,0.848158,0.342106,0.19908,0.143066,0.180084,0.265776,0.297163,0.273639,0.255711,0.265396,0.291774,0.341067,0.375838,0.338808,0.290364,0.226122,0.092644,0,0,0,0,0.018422,0.060105,0.061089,0.111233,0.308895,0.636594,0.886909,1,1,1,1,1,1,1,0.958641,0.48925,0,0.021737,0,0,0,0,0,0,0;0.035521,0.532816,0.696335,0.786471,0.897685,0.970342,0.944423,0.69952,0.755741,0.936686,0.666604,0.361362,0.296009,0.273467,0.312717,0.32987,0.312417,0.303291,0.310177,0.320866,0.319734,0.275357,0.174357,0.065065,0,0,0,0,0,0,0,0,0,0,0.011562,0.348155,0.702375,0.918567,0.948294,0.97011,0.986307,0.976295,0.884498,0.676374,0.214834,0,0,0,0,0,0,0,0,0,0;0,0.465252,0.658229,0.722285,0.812348,0.901578,0.886953,0.419447,0.410702,0.440707,0.519303,0.395037,0.391146,0.364607,0.349077,0.328199,0.284625,0.262408,0.241478,0.219285,0.174083,0.081609,0.006906,0,0,0,0,0,0,0,0,0,0,0,0,0,0.212893,0.515293,0.598354,0.563183,0.55048,0.498621,0.282947,0,0,0,0,0,0,0,0,0,0,0,0;0,0.314476,0.591495,0.660218,0.742509,0.830776,0.78317,0.386685,0.380619,0.365887,0.350121,0.357215,0.379002,0.384875,0.364006,0.323405,0.263861,0.196011,0.137649,0.071285,0,0.497727,0.285861,0.111187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.145165,0.077681,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.115909,0.483086,0.56168,0.652311,0.736101,0.823441,0.330364,0.351192,0.343205,0.308802,0.273856,0.254591,0.237245,0.253184,0.246584,0.204359,0.130436,0.030009,0,0,0,0.181375,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.375261,0.443229,0.520616,0.5919,0.657359,0.444525,0.209805,0.225163,0.189182,0.133825,0.088685,0.054661,0.054844,0.055099,0.040809,0,0,0,0,0,0,0.024536,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.291159,0.350819,0.40539,0.446629,0.480506,0.525521,0.386122,0.008115,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.209399,0.283899,0.328529,0.366004,0.381007,0.386917,0.407395,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.205487,0.278566,0.327642,0.356678,0.355682,0.347785,0.004852,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.099158,0.190684,0.274609,0.334937,0.36405,0.334534,0.081832,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.050713,0.153126,0.247488,0.13418,0.064396,0.071615,0.05508,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.100057,0.096701,0,0,0.016098,0.027506,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.017374,0.077709,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end

        case '4'; name='Hexacaudopteryx'; if isLoadCell; R=13;peaks=[1];mu=0.35;sigma=0.048;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0.018508,0.04998,0.068264,0.070585,0.057304,0.030339,0.008456,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.057199,0.171547,0.275713,0.34494,0.373693,0.367328,0.331646,0.271654,0.194928,0.111383,0.038996,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.002114,0.150464,0.376154,0.563301,0.680015,0.736206,0.744562,0.718479,0.669401,0.592237,0.487575,0.36243,0.232878,0.11749,0.030894,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.184104,0.535534,0.830993,1,1,1,1,1,0.968807,0.882939,0.76603,0.620889,0.463607,0.307582,0.170272,0.068924,0.008166,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.087307,0.582491,1,1,1,1,1,1,1,1,1,1,0.844938,0.667114,0.492575,0.336807,0.202685,0.093921,0.024723,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.469674,1,1,1,1,1,1,1,1,1,1,1,1,0.849458,0.66851,0.50792,0.362722,0.230911,0.127769,0.048484,0.001846,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.188553,0.866367,1,1,1,1,1,1,1,1,1,1,1,1,1,0.893427,0.722423,0.562879,0.414691,0.286686,0.175058,0.088265,0.027861,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.522539,0.994223,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.832969,0.662311,0.5046,0.369552,0.250557,0.155637,0.078754,0.021243,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.133704,0.594567,0.929063,1,1,1,1,1,0.949517,0.904391,0.909252,0.950366,0.984004,1,1,1,1,1,1,0.940436,0.775517,0.618256,0.474474,0.349834,0.241901,0.147868,0.063899,0.003913,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.061224,0.476062,0.872663,1,1,1,1,0.928157,0.756863,0.718753,0.74364,0.800039,0.851716,0.892327,0.905315,0.899418,0.896499,0.915896,0.960166,1,1,0.901948,0.739578,0.578205,0.446909,0.329919,0.216746,0.105333,0.021122,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.450778,0.855221,1,1,1,1,0.617289,0.480617,0.498136,0.550856,0.605761,0.630822,0.616681,0.557574,0.447248,0.297158,0.333608,0.536295,0.82357,0.984636,1,1,0.840977,0.663015,0.522811,0.395719,0.265183,0.132757,0.0294,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.486087,0.853786,1,1,1,0.467451,0.153938,0.174489,0.26443,0.340477,0.373913,0.351337,0.250543,0,0,0,0,0,0.389002,0.884268,1,1,1,0.893385,0.720916,0.584224,0.44266,0.289474,0.143539,0.035504,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.044999,0.518651,0.846448,0.997157,0.933945,0.262093,0.017012,0,0,0.005245,0.097943,0.099496,0,0,0,0,0,0,0,0.038199,0.765361,0.997278,1,1,1,0.946065,0.795783,0.644987,0.473759,0.301677,0.150488,0.045889,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.092249,0.520456,0.826101,0.598251,0.504972,0.27695,0.049463,0,0,0,0,0,0,0,0,0,0,0,0,0,0.706306,0.983834,1,1,1,1,1,0.895832,0.70141,0.496854,0.312657,0.163825,0.06371,0.005021,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.102994,0.496882,0.361295,0.477437,0.459981,0.315188,0.118288,0,0,0,0,0,0,0,0,0,0,0,0,0.129024,0.721621,0.976557,1,1,1,1,1,1,0.940366,0.725165,0.512502,0.332116,0.196274,0.089938,0.02246,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.087727,0.003258,0.252064,0.387635,0.426031,0.357932,0.203924,0.044183,0,0,0,0,0,0,0,0,0,0,0,0.388857,0.79434,0.975378,1,1,1,1,1,1,1,0.918821,0.731022,0.537928,0.375768,0.243727,0.136788,0.04562,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.172379,0.320545,0.393426,0.381181,0.274216,0.099535,0,0,0,0,0,0,0,0,0,0,0.172934,0.657803,0.907334,0.98042,0.991943,0.985524,0.959372,0.91561,0.881779,0.893786,0.944555,0.960202,0.896308,0.750225,0.583759,0.43388,0.310027,0.187629,0.080535,0.004371,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.128097,0.280141,0.372485,0.407617,0.323721,0.126773,0,0,0,0,0,0,0,0,0,0.21024,0.828925,1,1,0.989495,0.92489,0.791471,0.627274,0.506029,0.374926,0.414002,0.580842,0.795672,0.94149,0.930232,0.803242,0.649906,0.507405,0.374302,0.241023,0.103813,0.01269,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.110338,0.263027,0.384862,0.458729,0.38061,0.13474,0,0,0,0,0,0,0,0.053949,0.573571,1,1,1,1,0.995429,0.709261,0.373333,0.148054,0,0,0,0.201733,0.528113,0.871231,1,1,0.871972,0.713504,0.571057,0.425631,0.268177,0.114929,0.011753,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.104564,0.26267,0.428386,0.549548,0.443367,0.112584,0,0,0,0,0,0,0.15542,0.567436,1,1,1,1,1,0.936818,0.258018,0,0,0,0,0,0,0.304829,0.783643,0.989775,1,1,0.922912,0.763754,0.607684,0.442067,0.266915,0.101782,0.002251,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.01067,0.128551,0.2798,0.518079,0.696558,0.518827,0.071855,0,0,0,0,0,0.002329,0.513668,1,1,0.985497,1,1,1,0.459569,0,0,0,0,0,0,0,0.196122,0.716541,0.983247,1,1,1,0.940431,0.779705,0.606131,0.422982,0.237711,0.077953,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.184146,0.401246,0.710632,0.946891,0.607196,0.028218,0,0,0,0,0,0.107905,0.836437,1,0.966404,0.93503,0.998681,1,1,0,0,0,0,0,0,0,0,0.239806,0.715452,0.988314,1,1,1,1,0.942042,0.759705,0.564572,0.369096,0.184841,0.045363,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.171319,0.589132,0.95093,1,0.756716,0,0,0,0,0,0,0.390062,1,1,0.986439,0.987332,1,1,0.011128,0,0,0,0,0,0,0,0,0.306737,0.766404,0.998182,1,1,1,1,1,0.910119,0.7019,0.501012,0.309329,0.140988,0.025939,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.100915,0.711158,1,1,1,0,0,0,0,0,0,0.824723,1,1,1,1,0.900973,0.190865,0,0,0,0,0,0,0,0,0,0.406312,0.84142,1,1,1,1,1,1,1,0.840051,0.628273,0.431389,0.251022,0.099332,0.010879,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.399187,0.962217,1,1,1,0.634322,0.000068,0,0,0.266845,1,1,1,1,1,0.952008,0.262647,0,0,0,0,0,0,0,0,0,0.008419,0.582793,0.916349,1,1,1,1,1,1,1,0.95322,0.759836,0.56601,0.378389,0.211544,0.0775,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.50081,0.936015,1,1,1,1,0.887888,0.804055,0.846216,0.938773,1,1,1,1,1,0.877304,0.118587,0,0,0,0,0,0,0,0,0,0,0.347699,0.851547,0.975247,0.986794,0.979138,0.960687,0.933716,0.916849,0.920589,0.953097,0.972785,0.873071,0.715517,0.524323,0.343794,0.185247,0.061292,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.577793,0.902922,1,1,1,1,0.93824,0.864358,0.902773,0.96511,1,1,1,1,1,0.900419,0,0,0,0,0,0,0,0,0,0,0,0.355312,1,1,1,0.898856,0.765794,0.655522,0.583355,0.567206,0.614418,0.698049,0.83628,0.916049,0.864806,0.70167,0.507864,0.329076,0.170156,0.048492,0,0,0,0,0,0,0,0,0,0,0;0,0.352653,0.750136,1,1,1,0.816116,0.598109,0.661652,0.845865,0.965792,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0.011838,0.194919,0.700837,1,1,1,0.946052,0.540878,0.285711,0.165565,0.089347,0.017883,0.101638,0.308586,0.590488,0.86265,0.980416,0.893347,0.698361,0.497025,0.318382,0.159462,0.035052,0,0,0,0,0,0,0,0,0,0;0.044974,0.417138,0.862978,0.891755,0.492561,0.242126,0.080854,0.062843,0.377109,0.835083,1,1,1,1,1,1,0.241037,0,0,0,0,0,0,0,0,0.129717,0.577423,1,1,1,1,1,0.429476,0,0,0,0,0,0,0,0.347813,0.766513,0.986503,1,0.899421,0.674988,0.478903,0.301788,0.14014,0.022231,0,0,0,0,0,0,0,0,0;0.060057,0.452464,0.414135,0.589214,0.392571,0.179661,0.071274,0.02471,0.198694,0.84604,1,1,1,1,1,0.618088,0,0,0,0,0,0,0,0,0.295987,1,1,0.997057,0.988574,1,1,0.64104,0,0,0,0,0,0,0,0,0.180471,0.713597,0.977403,1,1,0.840315,0.631965,0.446711,0.265607,0.099464,0,0,0,0,0,0,0,0,0;0.058373,0.039339,0.365973,0.566745,0.4006,0.21243,0.108569,0.068688,0.195541,0.798537,1,1,1,1,0.745864,0,0,0,0,0,0,0,0,0.503268,1,1,0.934058,0.857089,0.934721,1,1,0.074864,0,0,0,0,0,0,0,0,0.153312,0.702203,0.982258,1,1,0.981471,0.763958,0.575302,0.392813,0.209384,0.05616,0,0,0,0,0,0,0,0;0.019052,0,0.273032,0.551943,0.491454,0.336078,0.235203,0.19145,0.413468,1,1,1,1,0.617222,0,0,0,0,0,0,0,0,0.576159,1,1,0.856934,0.651363,0.756276,0.968667,1,0.409408,0,0,0,0,0,0,0,0,0,0.307314,0.75588,0.994708,1,1,1,0.893942,0.700389,0.507411,0.316481,0.138937,0.018108,0,0,0,0,0,0,0;0,0,0.113866,0.467261,0.575097,0.520483,0.460886,0.444963,0.865859,1,1,1,0.474912,0,0,0,0,0,0,0,0,0.421889,1,1,0.847383,0.442489,0.534325,0.878964,1,0.824093,0.081755,0,0,0,0,0,0,0,0,0,0.438092,0.838514,1,1,1,1,1,0.844495,0.627982,0.420098,0.230302,0.074537,0,0,0,0,0,0,0;0,0,0,0.199915,0.435777,0.54322,0.604061,0.981004,1,1,1,0.653263,0,0,0,0,0,0,0,0,0.052731,1,1,1,0.329421,0.344967,0.801805,1,1,0.247132,0,0,0,0,0,0,0,0,0,0.180614,0.630356,0.904906,1,1,1,1,1,1,0.7611,0.529074,0.327412,0.15247,0.029876,0,0,0,0,0,0;0,0,0,0,0.009591,0.216809,0.822351,0.987073,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0.389615,0.274884,0.797049,1,1,0.457774,0,0,0,0,0,0,0,0,0,0.042375,0.57145,0.835846,0.938058,0.979879,1,1,1,1,1,0.908587,0.652497,0.431287,0.24391,0.093299,0.005139,0,0,0,0,0;0,0,0,0,0,0.292007,0.511921,0.723776,0.97,1,0.385307,0,0,0,0,0,0,0,0,0.770499,1,1,0.665792,0.38961,0.933509,1,1,0.717081,0,0,0,0,0,0,0,0,0,0.254782,0.922066,0.999366,0.973285,0.935906,0.90372,0.890889,0.89802,0.925931,0.964464,1,1,0.781842,0.549567,0.347097,0.180128,0.056975,0,0,0,0,0;0,0,0,0,0,0.131729,0.302329,0.54295,0.689794,0.602837,0.28505,0,0,0,0,0,0,0,0.37052,1,1,1,0.685491,1,1,1,0.818256,0,0,0,0,0,0,0,0,0.016383,0.843779,1,1,1,1,0.865876,0.687602,0.580866,0.543124,0.520134,0.709166,0.876299,0.989547,0.918876,0.683042,0.468032,0.289464,0.141784,0.029191,0,0,0,0;0,0,0,0,0,0.044448,0.24138,0,0.297516,0.51247,0.22031,0,0,0,0,0,0,0.110964,1,1,1,1,1,1,1,0.730626,0,0,0,0,0,0,0,0,0.176476,0.937225,1,1,1,1,1,0.623743,0.26571,0,0,0,0.06907,0.5791,0.913364,1,0.838086,0.601789,0.409199,0.246696,0.105118,0.007471,0,0,0;0,0,0,0,0,0,0,0,0.289193,0.436651,0.213807,0.008088,0,0,0,0,0.064698,1,1,1,1,1,1,1,0.537376,0,0,0,0,0,0,0,0,0.219025,0.931716,1,0.979944,0.980135,1,1,1,0.108192,0,0,0,0,0,0.219164,0.812464,1,1,0.755972,0.547571,0.37867,0.217177,0.067451,0,0,0;0,0,0,0,0,0,0,0,0.270858,0.380606,0.257081,0.089421,0.009438,0,0,0.292563,1,1,1,1,1,1,1,0.303225,0,0,0,0,0,0,0,0,0.080959,0.994295,1,0.954361,0.859018,0.913207,1,1,0.398051,0,0,0,0,0,0,0,0.73605,0.993164,1,0.920986,0.698802,0.525606,0.352521,0.165096,0.02251,0,0;0,0,0,0,0,0,0,0,0.208712,0.337718,0.32096,0.24215,0.186917,0.183179,0.746863,1,1,1,1,1,1,1,0.128035,0,0,0,0,0,0,0,0,0,0.952612,1,1,0.921881,0.881296,0.953845,1,0.932104,0.003358,0,0,0,0,0,0,0,0.706828,0.987553,1,1,0.863266,0.689319,0.508138,0.299214,0.088824,0,0;0,0,0,0,0,0,0,0,0.124844,0.25066,0.318469,0.345251,0.358575,1,1,1,0.995494,1,1,1,1,0.069103,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0.840203,0.232055,0,0,0,0,0,0,0,0.150245,0.71691,0.981034,1,1,1,0.869769,0.678101,0.44029,0.191455,0.016098,0;0,0,0,0,0,0,0,0,0.00856,0.1246,0.209175,0.261261,0.932759,0.913539,0.89104,0.882909,0.92331,0.993583,1,1,0.164949,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0.724572,0.337898,0.044435,0,0,0,0,0,0,0,0.357468,0.727651,0.967275,1,1,1,1,0.841002,0.586908,0.302469,0.061398,0;0,0,0,0,0,0,0,0,0,0,0.051262,0.725785,0.713116,0.696128,0.69681,0.750589,0.878602,1,1,0.442865,0,0,0,0,0,0,0,0,0.033577,0.978989,1,1,1,1,1,0.585603,0.193418,0.036677,0,0,0,0,0,0,0,0,0.062882,0.425086,0.71609,0.926054,1,1,1,1,0.986474,0.712533,0.405325,0.119589,0;0,0,0,0,0,0,0,0,0,0,0.511475,0.509198,0.502184,0.510903,0.563237,0.691829,0.88403,1,0.810372,0.00662,0,0,0,0,0,0,0,0,0.850245,1,1,1,1,1,0.028749,0,0,0,0,0,0,0,0,0,0,0,0.164615,0.442011,0.674797,0.871599,0.988419,1,1,1,1,0.813578,0.491649,0.171501,0;0,0,0,0,0,0,0,0,0,0.303472,0.307885,0.308891,0.325323,0.385214,0.508111,0.689938,0.912696,0.857525,0.232385,0,0,0,0,0,0,0,0,0.66418,0.954125,1,1,1,0.975903,0.688154,0,0,0,0,0,0,0,0,0,0,0,0,0.18421,0.418866,0.622302,0.809881,0.960639,1,1,1,1,0.89153,0.551172,0.213047,0.006704;0,0,0,0,0,0,0,0,0.080535,0.0822,0.083342,0.113327,0.195135,0.326978,0.499316,0.721328,0.367691,0.429681,0.24219,0.024431,0,0,0,0,0,0,0.406629,0.850401,0.976436,0.995731,0.987957,0.943518,0.807347,0.012764,0,0,0,0,0,0,0,0,0,0,0,0,0.144058,0.365847,0.575731,0.771218,0.937424,1,1,1,1,0.946589,0.585597,0.227082,0.009489;0,0,0,0,0,0,0,0,0,0,0,0,0.128197,0.315573,0.518185,0.161255,0.343432,0.428464,0.31661,0.122675,0.006932,0,0,0,0,0.139568,0.758117,0.944443,0.977773,0.967861,0.923595,0.823965,0.655554,0,0,0,0,0,0,0,0,0,0,0,0,0,0.047548,0.309744,0.556394,0.784602,0.959948,1,1,1,1,0.973093,0.585731,0.21497,0.003875;0,0,0,0,0,0,0,0,0,0,0,0,0.106576,0.319443,0,0.124044,0.304166,0.426902,0.429907,0.354193,0.250043,0.155569,0.077808,0.028342,0.34999,0.676668,0.956088,1,0.985587,0.930653,0.84106,0.728814,0.58317,0,0,0,0,0,0,0,0.033735,0.04188,0.021995,0,0,0,0,0.298125,0.638875,0.896951,1,1,1,1,1,0.966734,0.54732,0.173965,0;0,0,0,0,0,0,0,0,0,0,0,0,0.101578,0,0,0.056547,0.211158,0.351794,0.457585,0.540551,0.627075,0.715735,0.753192,1,1,0.950847,0.889489,0.903371,0.922674,0.878596,0.795313,0.71584,0.643018,0.358124,0,0,0,0,0.105444,0.215243,0.258865,0.241788,0.185516,0.096774,0.022895,0,0,0.539224,0.934535,1,1,1,1,1,1,0.899474,0.455961,0.102271,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.050425,0.162907,0.264051,0.362517,0.478169,0.641593,1,1,1,0.944371,0.640799,0.523129,0.558936,0.682205,0.806986,0.868527,0.889715,0.927849,0.920065,0.616351,0.533683,0.582986,0.586139,0.529405,0.46309,0.420155,0.372317,0.299604,0.210853,0.143786,0.320206,1,1,1,1,1,1,1,1,0.749299,0.307441,0.026594,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.144071,0.835718,1,1,0.80248,0.463883,0.31384,0.311835,0.560411,0.952543,1,1,1,1,1,1,1,0.78151,0.577869,0.471313,0.442846,0.455216,0.461716,0.456092,0.461547,1,1,1,1,1,1,1,1,1,0.505226,0.125475,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.025472,0.361599,0.811172,0.898556,0.758756,0.574023,0.434981,0.365478,0.35168,1,1,1,1,1,1,1,0.836744,0.575595,0.420896,0.348557,0.358593,0.418261,0.503692,0.585537,0.708035,1,1,1,1,1,1,1,1,0.619294,0.201018,0.002287,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.042275,0.268213,0.580964,0.794345,0.702036,0.583823,0.534954,0.571731,1,1,1,1,1,0.89047,0.69061,0.46306,0.290312,0.204589,0.182207,0.217477,0.299793,0.415764,0.54171,0.95348,0.962977,0.964282,0.96671,0.978589,0.998753,1,1,0.593707,0.1684,0.00143,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.28908,0.713244,0.797026,0.751012,0.73303,0.797499,1,1,1,0.950808,0.341045,0.210113,0.242517,0.199308,0.076779,0,0,0,0.068732,0.182182,0.595252,0.72046,0.714776,0.691238,0.670964,0.708529,0.825425,0.763851,0.390719,0.00618,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.398068,0.686522,0.789787,0.83468,0.911003,1,1,0.895131,0.096596,0,0,0,0,0,0,0,0,0,0.18503,0.310483,0.30036,0.253351,0.196095,0.186746,0.273979,0.374169,0.049875,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.208875,0.428628,0.570406,0.946358,0.88304,0.744139,0.160633,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.340732,0.365443,0.337412,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case '5'; name='Pyroscutium ambiguus'; if isLoadCell; R=13;peaks=[1];mu=0.349;sigma=0.0605;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0.008233,0.023311,0.033548,0.041409,0.040237,0.033159,0.025636,0.014927,0.00154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.002734,0.042702,0.098094,0.149072,0.193979,0.2213,0.232784,0.230648,0.216034,0.189269,0.162088,0.126767,0.090249,0.056845,0.027863,0.007819,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.035836,0.125269,0.22601,0.313303,0.388083,0.436588,0.464407,0.476095,0.474464,0.460634,0.434304,0.394034,0.35221,0.29657,0.241147,0.178482,0.126345,0.078782,0.037172,0.012528,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.066312,0.198933,0.339547,0.466424,0.569959,0.64115,0.690361,0.716897,0.723019,0.712382,0.688093,0.660499,0.627748,0.58406,0.540193,0.47667,0.409592,0.334111,0.257699,0.191257,0.131206,0.073944,0.024783,0,0,0,0,0,0,0;0,0,0,0,0.068434,0.238148,0.421863,0.595467,0.741475,0.84518,0.915394,0.950576,0.959862,0.948442,0.919241,0.876963,0.829732,0.790658,0.764899,0.73623,0.707102,0.652653,0.577677,0.494796,0.408635,0.325755,0.244082,0.160265,0.077236,0.01376,0,0,0,0,0;0,0,0,0.026512,0.222808,0.459379,0.69925,0.909993,1,1,1,1,1,1,1,0.996638,0.920921,0.862638,0.841414,0.846507,0.859183,0.853496,0.81703,0.744088,0.65075,0.553066,0.455359,0.355446,0.247212,0.131113,0.030039,0,0,0,0;0,0,0,0.156455,0.441488,0.759729,1,1,1,1,1,1,1,1,1,1,0.915393,0.834217,0.806364,0.839142,0.899557,0.965684,0.991205,0.959532,0.889047,0.793109,0.684869,0.568281,0.448472,0.313958,0.164557,0.037466,0,0,0;0,0,0.042183,0.35777,0.74456,1,1,1,1,1,1,1,1,1,1,0.963526,0.817874,0.715313,0.678754,0.726488,0.828177,0.964819,1,1,1,1,0.905171,0.782145,0.649429,0.509311,0.345485,0.165906,0.027118,0,0;0,0,0.217233,0.633892,1,1,1,1,1,1,0.969669,0.954788,0.952729,0.948525,0.933358,0.82933,0.667485,0.550931,0.505244,0.551663,0.671178,0.855183,1,1,1,1,1,0.978226,0.839683,0.696324,0.528987,0.332848,0.132104,0.006898,0;0,0.043095,0.442819,0.910499,0.995188,1,1,1,0.920516,0.692328,0.674173,0.75272,0.78595,0.802305,0.785441,0.683971,0.516179,0.390959,0.338317,0.374025,0.486615,0.678248,0.907036,0.986677,1,1,1,1,1,0.876466,0.705313,0.500569,0.265997,0.063386,0;0,0.216175,0.670148,0.863974,0.989664,1,1,0.775418,0.322124,0.220674,0.313932,0.493919,0.574581,0.624573,0.619189,0.554288,0.382745,0.248831,0.192697,0.222026,0.326315,0.505633,0.731407,0.861298,0.95436,1,1,1,1,1,0.882709,0.664686,0.402395,0.149191,0.004335;0,0.38418,0.6516,0.822067,0.983126,1,0.996646,0.181769,0,0,0.104354,0.269485,0.383397,0.466884,0.476228,0.39872,0.255407,0.107389,0.047338,0.079716,0.185737,0.360858,0.523326,0.670047,0.806756,0.923182,0.991882,1,1,1,1,0.830621,0.541843,0.242253,0.030971;0.109649,0.440676,0.573264,0.782373,0.972832,1,0.473854,0,0,0,0,0.090224,0.24333,0.376466,0.401894,0.287189,0.11947,0,0,0,0.040605,0.198459,0.32734,0.472436,0.630339,0.792901,0.938057,1,1,1,1,0.997253,0.676493,0.33213,0.065102;0.199227,0.34298,0.505197,0.732203,0.944654,1,0.177358,0.010937,0,0,0,0,0.229213,0.458875,0.4959,0.29815,0.010322,0,0,0,0,0.010969,0.14373,0.283771,0.455421,0.657248,0.874449,1,1,1,1,1,0.79696,0.41479,0.09682;0.162644,0.252555,0.437135,0.677215,0.905582,0.750228,0.240465,0.075024,0,0,0,0.074171,0.488792,0.769682,0.72377,0.430681,0.113742,0.003832,0,0,0,0,0,0.102364,0.278407,0.521696,0.83272,1,1,1,1,1,0.882012,0.477223,0.117924;0.049057,0.166934,0.374124,0.611568,0.835747,0.48073,0.301626,0.204847,0.117872,0.092264,0.131247,0.449333,0.856181,0.928944,0.702788,0.354995,0.28203,0.14717,0.014717,0,0,0,0,0,0.106323,0.401416,0.851723,1,1,1,1,1,0.904189,0.493404,0.115376;0,0.08566,0.301671,0.537178,0.707461,0.229243,0.293048,0.294673,0.280291,0.291513,0.345793,0.726206,0.872836,0.69292,0.322671,0.220585,0.263974,0.278468,0.168228,0.036625,0,0,0,0,0,0.342535,0.985909,1,1,1,1,1,0.836378,0.436174,0.078691;0,0.004276,0.225086,0.445474,0.377667,0.103705,0.200437,0.260704,0.294815,0.309179,0.434748,0.650353,0.579026,0.239655,0.017766,0.046662,0.137004,0.243473,0.285391,0.224018,0.103093,0.026655,0,0,0,0.429978,1,1,1,1,1,0.992978,0.674054,0.2863,0.011273;0,0,0.140698,0.342985,0.074601,0,0.052037,0.116253,0.149355,0.154257,0.305695,0.374515,0.180878,0,0,0,0,0.113449,0.240851,0.314549,0.30446,0.234343,0.174435,0.147182,0.174708,0.635052,1,1,1,0.984171,0.954112,0.756882,0.420054,0.051252,0;0,0,0.040642,0.227588,0,0,0,0,0,0,0.073574,0.036841,0,0,0,0,0,0,0.083662,0.226467,0.32633,0.362469,0.358349,0.360703,0.385683,0.668137,1,1,0.964074,0.873274,0.777152,0.525192,0.130804,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.03396,0.168471,0.266141,0.32592,0.363764,0.40827,0.456177,0.881538,0.928463,0.842864,0.705975,0.548689,0.335949,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.296739,0.327775,0.174607,0.21001,0.256521,0.286487,0.313928,0.782162,0.678822,0.513984,0.308043,0.123719,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.175762,0.349204,0.088533,0,0.033219,0.063314,0.065049,0.341058,0.490893,0.308491,0.068361,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.072734,0,0,0,0,0,0,0.292043,0.094953,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0895,0,0,0,0,0,0]; end
        case '6'; name='Aerogeminium ambiguus'; if isLoadCell; R=18;peaks=[1,5/6];mu=0.29;sigma=0.041;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0.001504,0.014509,0.024084,0.026071,0.018752,0.00872,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.068746,0.175479,0.270954,0.335071,0.368967,0.370431,0.34608,0.292901,0.221884,0.137122,0.055282,0.004475,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.021869,0.223522,0.4603,0.6445,0.762448,0.821305,0.839288,0.829242,0.793002,0.737115,0.649204,0.529538,0.381407,0.22549,0.085922,0.005886,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.030509,0.344972,0.731075,1,1,1,1,1,1,1,1,1,0.951505,0.807875,0.627845,0.422434,0.219991,0.059652,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.038453,0.427716,0.975655,1,1,1,1,1,1,1,1,1,1,1,1,1,0.829672,0.606295,0.368665,0.147366,0.01347,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.061308,0.499863,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.948032,0.744437,0.499775,0.247029,0.05733,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.073075,0.554732,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.837741,0.611415,0.358846,0.14173,0.020022,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.042254,0.56504,1,1,1,1,1,1,1,1,1,0.936652,0.842466,0.826564,0.874495,0.93734,0.979654,0.995773,1,1,1,1,1,0.917868,0.722575,0.493216,0.28058,0.129003,0.042753,0.004979,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.463872,1,1,1,1,1,1,1,1,1,0.860278,0.549409,0.388738,0.382,0.471233,0.601008,0.722491,0.804045,0.84957,0.88587,0.940762,0.994002,1,1,1,0.855704,0.664791,0.490874,0.350936,0.247948,0.167051,0.102076,0.052934,0.01834,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.180882,1,1,1,1,1,1,1,1,1,0.927465,0.337648,0,0,0,0,0.172241,0.339737,0.469163,0.559505,0.643597,0.756561,0.895513,0.99194,1,1,1,1,0.885523,0.770288,0.67215,0.579711,0.487114,0.395281,0.30549,0.210833,0.114136,0.026471,0,0,0,0,0,0,0,0,0;0,0,1,1,1,1,1,1,1,1,1,1,0.349257,0,0,0,0,0,0,0,0.139077,0.263756,0.387935,0.549394,0.746653,0.917502,0.993322,1,1,1,1,1,1,0.999428,0.917404,0.828319,0.728964,0.61127,0.467941,0.297913,0.127695,0.014235,0,0,0,0,0,0,0;0,0.141999,1,1,1,1,1,1,1,1,1,0.62362,0.01217,0.030518,0.046505,0.030842,0.000867,0,0,0,0,0.015982,0.177704,0.3726,0.597144,0.786493,0.894464,0.947378,0.976702,0.995128,1,1,1,1,1,1,1,1,0.857078,0.676257,0.460289,0.229515,0.050922,0,0,0,0,0,0;0,0.851465,0.98059,0.995745,1,1,1,1,1,1,1,0.000429,0.110278,0.309229,0.41529,0.39187,0.270056,0.087654,0,0,0,0,0.00116,0.230536,0.447696,0.60157,0.684202,0.729461,0.770299,0.829892,0.906648,0.973585,1,1,1,1,1,1,1,1,0.830126,0.571734,0.299656,0.080005,0,0,0,0,0;0,0.704395,0.835866,0.868041,0.91342,0.994492,1,1,1,1,0.105421,0,0.433382,0.830993,0.821055,0.729151,0.634536,0.369613,0.040864,0,0,0,0,0.094187,0.288794,0.369793,0.390472,0.402044,0.438569,0.511353,0.630403,0.776195,0.899771,0.974019,1,1,1,1,1,1,1,0.936184,0.635286,0.326859,0.085261,0,0,0,0;0.140295,0.46425,0.568351,0.605808,0.721852,0.932344,1,1,1,0.610403,0,0.015783,0.988054,1,1,0.943291,0.968461,0.595563,0.049112,0,0,0,0,0,0.127047,0.112694,0.05654,0.041147,0.068654,0.147165,0.274425,0.439879,0.607723,0.748054,0.861018,0.95082,1,1,1,1,1,1,0.992857,0.646625,0.307124,0.061869,0,0,0;0,0.145553,0.225892,0.29831,0.497093,0.835589,1,1,1,0,0,0.28912,1,1,1,1,1,0.506117,0,0,0,0,0,0,0.021228,0,0,0,0,0,0,0.040621,0.20157,0.346968,0.486021,0.645534,0.833622,0.978053,1,1,1,1,1,0.982274,0.592466,0.239961,0.022525,0,0;0,0,0,0,0.281151,0.716911,1,1,0.120951,0,0,0.705362,0.995099,1,1,1,0.981571,0,0,0,0,0,0,0.042278,0.049678,0,0,0,0,0,0,0,0,0,0.024886,0.195528,0.442039,0.752831,0.976206,1,1,1,1,1,0.89321,0.474206,0.136866,0,0;0,0,0,0,0.077669,0.573423,0.971456,1,0.214855,0,0,0.625194,0.921412,0.982309,0.950011,0.798035,0,0,0,0,0,0,0.012991,0.388349,0.229296,0,0,0,0,0,0,0,0,0,0,0,0,0.374203,0.796271,0.998445,1,1,1,1,1,0.729272,0.308647,0.038365,0;0,0,0,0,0,0.406086,0.876421,0.840179,0.418808,0.099442,0.052939,0.596465,0.817618,0.792041,0.563543,0,0,0,0,0,0,0,0.377755,0.92934,0.456162,0,0,0,0,0,0,0,0,0,0,0,0,0,0.51906,0.920888,1,1,1,1,1,0.991439,0.504254,0.133599,0;0,0,0,0,0,0.221584,0.7049,0.743827,0.641797,0.397924,0.535546,0.770106,0.786981,0.494415,0,0,0,0,0,0,0,0,1,1,0.501889,0,0,0,0,0,0,0.006648,0,0,0,0,0,0,0.247437,0.804347,0.999315,1,1,1,1,1,0.707608,0.250152,0.010355;0,0,0,0,0,0.021069,0.250694,0.541121,0.703352,0.79621,0.992404,1,0.875032,0.250403,0.015754,0,0,0,0,0,0,1,1,1,0.114949,0,0,0,0,0,0.084867,0.149819,0.067931,0.00884,0,0,0,0,0.113153,0.749873,0.99531,1,1,1,1,1,0.904326,0.368859,0.044308;0,0,0,0,0,0,0,0.269602,0.676657,0.976681,0.997879,1,0.81884,0.473811,0.300777,0.140812,0.037203,0.00937,0.0147,0.06989,0.689634,1,1,0.965474,0,0,0,0,0,0,0.184929,0.313805,0.250053,0.181717,0.130448,0.101526,0.095321,0.102748,0.258752,0.843661,1,1,1,1,1,1,1,0.465593,0.072343;0,0,0,0,0,0,0,0.157621,0.613945,0.775975,0.814921,0.829931,0.473677,0.492249,0.496992,0.468976,0.418634,0.390788,0.444536,0.602686,1,1,1,0.626197,0,0,0,0,0,0,0.144821,0.305163,0.342583,0.39409,0.421508,0.432888,0.446265,0.45032,0.659404,1,1,1,1,1,1,1,1,0.523396,0.079148;0,0,0,0,0,0,0,0.038411,0.343251,0.439792,0.452204,0.144507,0.088774,0.135135,0.20661,0.287104,0.36448,0.436456,0.51848,0.885298,0.932155,1,1,0,0,0,0,0,0,0,0,0.051522,0.187755,0.453288,0.671133,0.792053,0.848312,0.848979,1,1,1,1,1,1,1,1,1,0.52032,0.048607;0,0,0,0,0,0,0,0,0.022705,0.063807,0,0,0,0,0,0,0,0.04835,0.414862,0.462114,0.649791,0.969786,1,0.061315,0,0,0,0,0,0,0,0,0,0.206798,0.715338,1,1,1,1,1,1,1,1,1,1,1,1,0.443757,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.042884,0.38868,0.890195,0.682928,0.270553,0.052114,0.002974,0,0,0,0,0,0,0,0,0.481815,0.996082,1,1,1,1,1,1,1,1,1,1,1,0.283068,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.199178,0.596988,0.701486,0.55324,0.425432,0.375925,0.370549,0.367853,0.317192,0.161247,0.009959,0,0,0,0.112749,0.856103,1,1,1,1,1,1,1,1,1,1,0.88897,0.059136,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.046241,0.371851,0.568254,0.580744,0.529235,0.484342,0.485273,0.529682,0.570382,0.500688,0.227462,0.002488,0,0,0,0.664255,0.980675,1,1,1,1,1,1,1,1,0.942928,0.641907,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.060733,0.246344,0.250643,0.169922,0.111485,0.1345,0.220613,0.361903,0.528677,0.498843,0.174821,0,0,0,0.609383,1,1,1,1,1,1,1,1,1,0.674165,0.318894,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.047788,0.318175,0.548057,0.457307,0.185043,0.053305,0.030275,0.923907,1,1,1,1,1,1,1,1,0.862986,0.28306,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.100591,0.442965,0.581754,0.510049,0.416488,0.675207,1,1,1,1,1,1,1,1,0.935911,0.480045,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.279557,0.495275,0.593734,0.648135,1,1,1,1,1,1,1,0.993262,0.911605,0.637947,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.068257,0.258713,0.389366,0.516177,0.862257,0.880239,0.897843,0.922029,0.943789,0.945181,0.910422,0.810891,0.603615,0.155199,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.051516,0.468064,0.522696,0.569108,0.624269,0.682762,0.726305,0.717817,0.636797,0.461396,0.155659,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05792,0.14198,0.223181,0.313137,0.395093,0.434389,0.395317,0.261002,0.020634,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.082012,0.095545,0.012201,0,0,0,0,0,0,0,0,0]; end

        case '7'; name='Kronium vagus'; if isLoadCell; R=18;peaks=[1,1/3];mu=0.22;sigma=0.025;dt=0.1;cells=[0,0,0,0,0,0,0,0,0.027476,0.077167,0.114061,0.134954,0.117971,0.078477,0.022158,0,0,0,0,0,0,0;0,0,0,0,0,0.002793,0.078454,0.189615,0.305937,0.396977,0.460669,0.480279,0.469385,0.421478,0.333435,0.183753,0.034148,0,0,0,0,0;0,0,0,0,0.058539,0.210466,0.377092,0.52701,0.640697,0.710823,0.735617,0.718462,0.690539,0.670463,0.640702,0.54674,0.343833,0.101092,0,0,0,0;0,0,0,0.107882,0.323362,0.537401,0.708082,0.828484,0.916357,0.970306,0.973665,0.93385,0.868123,0.827057,0.822445,0.818449,0.699536,0.420436,0.112768,0,0,0;0,0,0.117452,0.391676,0.662625,0.866648,0.989142,1,1,1,1,1,1,0.967508,0.970237,1,1,0.784001,0.394701,0.068086,0,0;0,0.077062,0.391232,0.741437,1,1,1,0.959732,0.916117,0.89392,0.883953,0.871768,0.865241,0.8868,0.954813,1,1,1,0.727196,0.262561,0.009087,0;0.005752,0.288533,0.71812,1,1,0.994743,0.803954,0.62312,0.549167,0.542557,0.559745,0.572999,0.580549,0.608827,0.698152,0.895105,1,1,1,0.512766,0.092438,0;0.08249,0.511165,1,1,1,0.793178,0.428772,0.238266,0.155994,0.158264,0.20494,0.25315,0.283526,0.317371,0.388248,0.574203,0.92587,1,1,0.783728,0.222712,0;0.183352,0.700242,1,1,0.999803,0.454882,0.061191,0,0,0,0,0,0,0,0.074614,0.244446,0.599857,1,1,1,0.365117,0.006254;0.328535,0.780034,1,1,0.902449,0.125662,0,0,0,0,0.039912,0.03217,0,0,0,0,0.243036,0.915183,1,1,0.494698,0.027799;0.438241,0.753418,1,1,0.694247,0,0,0,0,0.236903,0.275767,0.278569,0.171554,0,0,0,0,0.687115,1,1,0.601936,0.044424;0.460384,0.594816,1,1,0.473872,0,0,0,0.304248,0.200141,0,0.036601,0.297122,0.090451,0,0,0,0.499194,1,1,0.675646,0.043465;0.245828,0.35602,1,1,0.251556,0,0,0.088715,0.493414,0,0,0,0.070038,0.39057,0,0,0,0.430783,1,1,0.67691,0.020761;0,0.073316,1,1,0,0,0,0.491339,0.320293,0,0,0,0,0.690038,0,0,0,0.564924,1,1,0.574595,0;0,0,1,1,0,0,0,0.897765,0.100642,0,0,0,0,1,0,0,0,0.948756,1,0.980744,0.368909,0;0,0.136608,1,1,0.002918,0,0,0.887484,0.383271,0.024432,0,0,0.218831,1,0,0,0.002784,1,1,0.831545,0.099434,0;0,0.037172,0.922255,0.824183,0.118601,0,0,0.661934,0.441869,0.319951,0.241135,0.307747,0.400144,0.557022,0,0,0.212967,1,0.982048,0.571565,0,0;0,0,0.740872,0.829117,0.365161,0.078252,0,0.537637,0.391412,0.400731,0.335692,0.19374,0.637522,0,0,0.127282,0.644021,1,0.772691,0.250663,0,0;0,0,0.466953,0.691936,0.586351,0.371233,0.21252,0.419806,0.522976,0.41278,0.174911,0,0.353537,0,0.08806,0.498811,0.928216,0.870382,0.352928,0,0,0;0,0,0.060141,0.404896,0.590984,0.566751,0.910309,0.73498,0.856408,0.616164,0.121235,0.015144,0.062268,0.141114,0.463464,0.772813,0.873725,0.489998,0,0,0,0;0,0,0,0.06449,0.26827,0.537569,0.878184,0.938809,0.975047,0.710225,0.373399,0.477521,0.349905,0.505903,0.65855,0.595983,0.492455,0.004668,0,0,0,0;0,0,0,0,0,0.386814,0.507575,0.619474,0.736098,0.490282,0.39753,0.714648,0.464879,0.462897,0.384083,0.373887,0.020102,0,0,0,0,0;0,0,0,0,0,0,0.103441,0.251973,0.389773,0.137745,0.569237,0.183165,0.17707,0.146999,0.043991,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.16259,0,0,0,0,0,0,0,0,0,0,0]; end
        case '8'; name='Crucium arcus vagus'; if isLoadCell; R=27;peaks=[1,2/3,1/3];mu=0.22;sigma=0.023;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.015527,0.021849,0.014244,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.069998,0.199579,0.31507,0.382258,0.410275,0.390542,0.328264,0.219834,0.094862,0.010494,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.056286,0.247147,0.440847,0.577245,0.636606,0.637114,0.617429,0.607292,0.61104,0.608628,0.575951,0.48523,0.344736,0.170931,0.034711,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.19345,0.555283,0.850398,1,1,1,1,1,0.924713,0.791861,0.735536,0.725474,0.720018,0.68286,0.611331,0.507097,0.358972,0.17315,0.023078,0,0,0,0,0,0,0;0,0,0,0,0,0.095702,0.576605,0.936755,1,0.998256,0.975653,0.977271,0.99466,1,1,1,1,1,0.944,0.888165,0.835256,0.752476,0.654222,0.562088,0.457737,0.29033,0.083728,0,0,0,0,0,0;0,0,0,0,0.199217,0.78875,1,0.980497,0.867133,0.811216,0.841873,0.927106,0.994436,1,1,1,1,1,1,0.965652,0.926106,0.860409,0.757706,0.652864,0.566804,0.490545,0.350832,0.110521,0,0,0,0,0;0,0,0,0.17934,0.935156,1,0.991752,0.887529,0.825298,0.833182,0.894452,0.968746,1,1,1,1,1,1,1,1,0.848041,0.842107,0.798371,0.726824,0.643786,0.570766,0.509138,0.355782,0.081817,0,0,0,0;0,0,0.057536,0.841304,1,1,0.998893,0.95374,0.929112,0.932875,0.954403,0.978181,0.989902,0.98954,0.974727,0.946321,0.917931,0.909345,0.932141,0.969707,1,1,1,0.7922,0.736161,0.673534,0.608231,0.53152,0.317421,0.022467,0,0,0;0,0,0.423422,1,1,1,1,1,0.989919,0.96946,0.955044,0.93019,0.871768,0.783429,0.701072,0.647067,0.625645,0.629134,0.676245,0.763475,0.884128,0.983498,1,1,1,1,0.809518,0.693613,0.560039,0.227874,0,0,0;0,0.04632,0.595417,1,1,1,1,1,1,0.967302,0.904719,0.782105,0.529384,0.218678,0.057135,0.048859,0.112975,0.191732,0.289214,0.42342,0.614206,0.836777,0.985463,1,1,1,1,1,0.801589,0.554266,0.094599,0,0;0,0.103308,0.506585,0.810095,0.932262,1,1,1,0.993503,0.908406,0.760385,0.442313,0,0,0,0,0,0,0,0,0,0.361032,0.852615,0.998056,1,1,1,1,1,0.84008,0.431064,0,0;0,0.063182,0.35649,0.648904,1,1,1,1,0.897389,0.721264,0.346311,0,0,0,0,0,0,0,0,0,0,0,0,0.888756,1,1,1,1,0.981485,0.986004,0.767459,0.155656,0;0,0,0.228631,1,1,1,1,0.849267,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.510829,0.923112,1,1,0.993818,0.931702,0.954831,0.927541,0.485962,0;0,0,0.558975,1,1,1,0.103324,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.038008,0.715325,0.952256,0.993799,0.986601,0.924006,0.899282,0.997279,0.747061,0.055326;0,0,1,1,1,0,0,0,0,0,0,0,0,0,0.348757,1,1,1,0.462037,0,0,0,0,0,0.508228,0.872513,0.978765,0.99284,0.968224,0.928822,0.990306,0.916734,0.220272;0,0.593526,1,1,0.137,0,0,0,0,0,0,0,0,0.080647,0.912082,0.956725,0.99544,1,1,0.765126,0,0,0,0,0.387873,0.81034,0.978609,1,1,0.998306,1,1,0.360003;0,0.731421,1,0.731479,0,0,0,0,0,0.439074,0.173812,0.046876,0,0.503393,0.567698,0.656397,0.789603,0.92666,1,1,0,0,0,0,0.281806,0.767331,0.991764,1,1,1,1,1,0.447905;0.239349,0.783396,0.990447,0.482604,0,0,0,0.544422,1,1,1,0.679383,0,0,0,0.161895,0.368415,0.602929,0.865502,0.937352,0,0,0,0,0,0.709976,0.99949,1,1,1,1,1,0.478598;0.356846,0.789043,0.934108,0.346331,0,0,0.225973,0.998771,1,1,1,0.075315,0,0,0,0,0,0,0,0.212646,0.006832,0,0,0,0,0.483446,1,1,1,1,1,1,0.448817;0.375458,0.764406,0.953267,0.282319,0,0,0.413771,1,1,1,1,0.105068,0,0,0,0,0,0,0,0,0.213476,0.010418,0,0,0,0,1,1,1,1,1,0.880828,0.358225;0.328296,0.710055,1,0.726345,0,0,0.341099,0.99312,1,1,0.944445,0.206566,0,0,0,0,0,0,0,0.049702,0.655058,0.31449,0,0,0,0,0.8948,1,1,1,0.697695,0.611081,0.219044;0.230421,0.609768,0.964748,1,0,0,0.104715,0.82634,1,1,0.936181,0.422331,0,0,0,0,0,0,0,0.524928,1,1,0,0,0,0,0.303042,1,1,1,0.374864,0.323206,0.06692;0.120866,0.458552,0.859046,1,0.004938,0,0,0,1,1,0.98102,0.800044,0.008204,0,0,0,0,0,0.223711,1,1,1,1,0,0,0,0.077024,1,1,1,0.105514,0.076073,0;0.033507,0.277256,0.674988,1,0.124516,0,0,0,0.111131,1,1,1,0.421736,0,0,0,0,0.261742,1,1,1,1,0.983511,0,0,0,0.134878,1,1,0.916213,0,0,0;0,0.111415,0.443037,0.872885,0.411757,0,0,0,0,1,1,1,1,0.530061,0.173943,0.180338,0.504479,1,1,1,1,1,0.803168,0,0,0,0.438429,1,0.969897,0.519679,0,0,0;0,0,0.212985,0.621169,0.941189,0.185615,0,0,0,0.847018,1,1,1,1,1,1,1,1,1,1,1,0.980453,0.42923,0,0,0.006078,0.940013,1,0.749538,0,0,0,0;0,0,0,0.323945,0.768142,0.569167,0.129727,0,0,0.724651,1,1,1,1,1,1,1,1,1,1,0.854844,0.432863,0,0,0,0.305205,1,0.870273,0.407505,0,0,0,0;0,0,0,0.016713,0.42087,0.860576,0.687802,0.167716,0.003202,0.365141,0.890658,1,1,1,1,1,0.118928,0,0,0,0,0,0,0,0.163236,0.890507,0.945748,0.558356,0.00093,0,0,0,0;0,0,0,0,0.058706,0.51385,0.926099,0.745487,0.221339,0.051451,0.365388,0.743567,0.990165,1,0.869703,0,0,0,0,0,0,0,0,0.208313,0.784188,0.963798,0.630336,0.168371,0,0,0,0,0;0,0,0,0,0,0.166248,0.627121,0.840347,0.515964,0.268178,0.126146,0.07398,0.092116,0.024763,0,0,0,0,0,0,0.011131,0.139181,0.518987,1,0.926254,0.623505,0.241613,0,0,0,0,0,0;0,0,0,0,0,0,0.32733,0.371275,0.533174,0.485872,0.367042,0.276385,0.228567,0.198597,0.164645,0.125875,0.105156,0.11934,0.190402,0.322727,0.979518,1,0.957469,0.780134,0.519912,0.232657,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.143934,0.345932,0.452415,0.489296,0.500037,0.518466,0.560911,0.621793,0.674282,0.704289,0.720842,0.997267,0.945585,0.84832,0.702713,0.515974,0.322505,0.1313,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.041725,0.284536,0.352734,0.39279,0.454251,0.51265,0.54361,0.541047,0.548784,0.652152,0.600655,0.539573,0.445309,0.302334,0.143241,0.02909,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.115719,0.31887,0.412263,0.429873,0.384355,0.29264,0.181694,0.069934,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.022129,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.11065,0.170485,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.066772,0.138305,0.17233,0.180387,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.104636,0.006061,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case '9'; name='Astrium currens'; if isLoadCell; R=18;peaks=[1,1/3,2/3,1];mu=0.2;sigma=0.022;dt=0.1;cells=[0,0,0,0,0,0,0,0.490424,1,1,1,0.733754,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.582628,1,1,1,1,1,0.047957,0,0,0,0,0,0,0,0,0.034294,0.042675,0,0,0,0,0;0,0,0,0,0,0,1,1,1,1,1,1,0.342889,0.09101,0.191559,0.407172,0.507653,0.472116,0.18573,0.007664,0.097697,0.826709,1,0.760937,0,0,0,0;0,0,0,0,0,0.157013,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.635957,1,1,1,1,0,0,0;0,0,0,0,0,0.274489,1,1,1,0.638612,0.326433,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.78127,0,0;0,0,0,0,0,0.234578,1,1,1,0.60303,0.025383,0.712511,0.628495,0.331806,0.951422,1,1,1,0.933566,0.478326,1,1,1,1,1,1,0,0;0,0,0,0,0,0.419699,1,1,1,1,1,1,0.888625,0,0,0.048049,0.93749,1,0.891417,0,0,0.943579,1,1,1,1,0.731461,0;0,0,0,0,0,0.94272,1,1,0.623905,0.957718,1,1,0.969078,0,0,0,0,0,1,1,0.376333,0.048906,1,1,0.881857,0.878529,0.684283,0;0,0,0,0,0.192068,1,1,0.989664,0.117409,0,0.714653,1,0.73429,0,0,0,0,0,1,1,1,0,0,1,0.45209,0,0.214755,0;0,0,0,0.014687,1,1,1,1,0,0,0,0,0,0.029278,0.524095,0.719328,0,0,0,1,1,0,0,1,0.397208,0,0,0;0,0,0,0.588229,1,1,1,0.962353,0,0,0,0,0,0,1,1,0.391893,0,0,0.000476,0.996128,0,0,0.945903,0.27963,0,0,0;0,0,0,1,1,1,1,0.512093,0,0,0.103137,1,0,0,1,1,0.526534,0,0,0.003209,0.731103,0,0,1,0,0,0,0;0,0.06504,0.044624,1,1,0.763371,0.971946,0.018576,0,0,1,1,1,0,0,0.567632,0.445758,0,0.000611,0.247358,0.491701,0,0,1,0,0,0,0;0.740656,1,0.41633,1,0.879587,0.235077,1,1,0,0,0,1,1,0,0,0.156192,0.691981,0,0,0.88202,0.862271,0,0,1,0,0,0,0;1,1,1,1,0.417367,0.003188,1,1,0,0,0,0.473006,0.970471,0,0,1,0.541194,0,0,0,0.563768,0,0,0.254968,1,0,0,0;1,1,1,1,0.172646,0,1,1,1,0,0,0,0.663057,0.059678,0,0.672421,0.274722,0,0,0,0,0.492186,0,0.083117,1,0,0,0;1,1,1,1,0.881189,0,0,1,1,0,0,0.003112,0,0,0,0,0,0,0,0,0.405342,1,0.491497,0.222862,1,1,0,0;1,1,0.995862,0.892344,1,0,0,0.370026,0.880041,0,0.005764,0.33011,0,0,0,0,0.398057,0,0,0,0.672196,1,1,0.64067,1,1,0.537037,0;0,1,0.843464,0,0.547133,1,0,0,0,0,0,0.358462,0,0,0,0,0.84086,0.737231,0,0,0,1,1,0.965554,1,1,1,0.328443;0,0,0.835317,0,0,0.57411,1,0.555026,0.01048,0,0,0,0,0,0,0,0,0,0,0,0,0,0.657222,1,1,1,0.936582,0;0,0,0,0,0,0,0.434109,0.867349,1,0.45579,0,0,0.420931,0.680762,0.072777,0,0,0,0,0,0.169198,0.55919,1,1,0.978834,0.830446,0.481692,0;0,0,0,0,0,0,0,0,0.494748,1,0.14811,0,0.688293,1,1,0,0,0,0.217221,0.871645,0.999913,0.903526,0.685058,0.415428,0.266848,0.176176,0,0;0,0,0,0,0,0,0,0.282493,0,1,1,0.268341,0.60076,1,1,0,0.027403,0.577214,0.889409,0.265325,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.396144,0.410018,0.128891,1,1,0.991226,1,1,1,1,0.836806,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.833592,1,1,1,1,1,0.794279,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0.740183,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.83722,0.988878,0.982994,0.830174,0.332431,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.448837,0.491953,0.040106,0,0,0,0,0,0,0,0,0,0,0,0,0]; end

        case 'S+1'; name='Quadrium vagus'; if isLoadCell; R=36;peaks=[1,2/3,1/3,2/3];mu=0.13;sigma=0.007;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.03,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.55,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.99,0.66,0.47,0.38,0.19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.31,0.28,0.4,0.5,0.27,0,0.04,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.55,1,0.23,0,0.06,0.24,0.52,0.59,0.2,0.35,0.07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.08,0.99,0.34,0,0,0,0.19,0.65,1,0.88,0.44,0.13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.97,0.48,0,0,0,0,0.07,0.89,1,0.75,0.21,0,0.02,0.06,0.06,0.03,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.29,0,0,0,0,0,0,0.84,1,0.62,0,0,0,0.53,0.6,0.11,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.03,0,0,0,0,0,0,0,0.49,1,0.84,0.32,0.18,0.62,1,1,0.66,0.41,0.17,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.76,0.7,0,0,0,0,0,0,0,0,0.08,1,1,1,1,1,1,0.36,0,0.05,0.22,0.4,0.29,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.02,0.76,0.08,0,0,0,0,0,0,0,0,0.85,1,1,1,1,0.76,0,0,0,0,0,0,0.24,0.48,0.06,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.2,0.91,0,0,0,0,0,0,0,0,0.63,1,1,1,1,0.65,0,0,0,0,0,0,0,0,0.19,0.52,0.05,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.55,0.71,0,0,0,0,0,0,0,0.8,1,1,0.83,1,1,0.21,0,0,0,0,0,0,0,0,0,0.29,0.53,0.5,0.15,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0.51,1,1,0.84,0.55,0.7,1,0.92,0,0,0,0,0,0,0,0,0,0.64,1,1,0.7,0.33,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0.36,1,1,1,1,1,1,1,0.72,0,0,0,0,0,0,0,0.17,0.95,1,1,1,1,0.48,0.29,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.16,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0.97,0.48,0,0,0,0,0,0,0,0,1,1,0.87,0.81,1,0.8,0,0.04,0.25,0.25,0,0,0,0,0,0,0,0,0;0,0,0,0,0.61,0.8,0.19,0,0,0,0,0,0,1,1,1,1,1,0.44,0,0,0,0,0,0,0,0,0,0.6,1,0.9,0.2,0.26,0.72,0.71,0.05,0,0,0,0.18,0.14,0,0,0,0,0,0,0,0;0,0,0.18,0.53,0.24,0.37,0,0,0,0,0,0.63,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0.74,1,1,0.57,0,0.09,0.72,0.86,0,0,0,0,0,0,0.28,0,0,0,0,0,0,0,0;0,0.16,0.44,0.31,0.56,0.29,0.09,0,0,0.28,1,1,1,1,1,0.97,0.1,0,0,0,0,0,0,0,0.47,1,1,0.59,0,0,0.46,0.89,1,0.01,0,0,0,0,0,0,0.2,0.23,0,0,0,0,0,0,0;0.03,0.34,0.45,0.32,0.6,0.44,0.41,0.54,0.99,1,1,0.95,0.87,1,0.99,0.31,0,0,0,0,0,0,0,1,1,0.32,0.16,0.11,0.25,0.99,1,0.85,0,0,0,0,0,0,0,0,0.22,0.37,0,0,0,0,0,0,0;0.14,0.4,0.59,0.66,1,1,1,1,1,1,0.93,0.95,1,1,0,0,0,0,0,0,0,0.23,0.83,0.61,0,0,0.11,0.63,1,0.54,0.15,0,0,0,0,0,0,0,0,0,0.38,0.45,0,0,0,0,0,0,0;0.13,0.38,0.74,0.96,1,1,1,1,1,1,1,1,0.57,0,0,0,0,0,0,0,0.46,0.93,0.35,0,0,0.01,0.56,0.98,0,0,0,0,0,0,0,0,0.44,0.46,0.69,0.94,1,0.91,0.68,0.19,0,0,0,0,0;0.02,0.15,0.47,0.86,1,1,1,1,1,1,0.51,0,0,0,0,0,0,0,0,1,0.8,0.37,0.08,0,0,0.84,0.91,0,0,0,0,0,0,0.19,0.87,0.97,0.69,0.43,0.67,1,1,1,0.6,0.6,0.28,0,0,0,0;0,0,0.07,0.6,0.98,1,0.98,0.75,0.5,0.09,0,0,0,0,0,0,0,0.98,1,0.84,0.5,0.41,0.45,0.72,1,0.74,0,0,0,0,0,0,0.39,0.97,0.73,0.21,0.15,0.24,0.45,1,1,0.12,0.14,0.29,0.49,0.07,0,0,0;0,0,0,0.49,0.88,0.67,0.29,0.17,0.13,0,0,0,0,0,0,0.16,1,0.96,0.37,0.14,0.78,0.95,0.55,0.24,0,0,0,0,0,0,0,0.28,1,0.59,0,0,0.05,0.52,1,0.91,0,0,0,0.03,0.35,0.2,0,0,0;0,0,0,0.48,0.61,0.24,0,0.03,0.11,0,0,0,0,0,0.21,0.85,0.67,0.52,0.52,1,1,0,0,0,0,0,0,0,0,0.79,1,1,0,0,0,0.61,1,1,0.13,0,0,0,0,0,0.32,0.28,0,0,0;0,0,0.01,0.46,0.4,0.1,0,0.07,0.19,0.03,0,0,0,0.14,0.52,0.49,0.88,1,1,0.31,0,0,0,0,0,0,0,0.39,1,0.78,0.26,0,0,0.2,1,1,0.11,0,0,0,0,0,0,0.01,0.48,0.38,0.24,0,0;0,0,0.02,0.44,0.35,0.13,0.1,0.33,0.34,0.24,0.12,0.05,0.12,0.37,0.55,0.72,0.88,1,0,0,0,0,0,0,0,0,0.32,1,0.04,0,0,0.02,0.62,1,0.22,0,0,0,0,0,0,0,0,0.34,1,0.57,0.05,0,0;0,0,0.03,0.45,0.45,0.32,0.43,0.71,0.73,0.74,0.71,0.68,0.73,0.73,0.61,0.56,0.36,0,0,0,0,0,0,0,0,0.24,1,0,0,0,0.02,0.98,1,0,0,0,0,0,0,0,0,0,1,1,0.74,0.22,0.04,0,0;0,0,0.05,0.47,0.63,0.69,1,1,1,1,1,0.91,0.66,0.42,0.16,0,0,0,0,0,0,0,0,0,0.21,0.96,0.36,0,0,0.09,1,0.98,0,0,0,0,0,0,0,0,0,1,1,0.6,0.35,0.29,0.29,0.28,0.21;0,0,0.06,0.48,0.8,1,1,1,1,1,0.99,0.5,0.11,0,0,0,0,0,0,0,0,0,0,1,0.98,0.59,0.33,0.32,0.82,1,0.39,0,0,0,0,0,0,0,0,0.82,1,1,0.57,0.54,0.63,0.73,0.25,0.23,0.05;0,0,0.05,0.41,0.79,1,1,1,1,1,0.47,0.13,0,0,0,0,0,0,0,0,0,0.95,1,1,0.47,0.67,0.83,0.87,0.4,0,0,0,0,0,0,0,0,0.75,1,1,1,1,1,1,0.86,0.69,0.67,0.61,0.34;0,0,0.01,0.2,0.48,0.75,0.94,1,0.7,0.1,0,0,0,0,0,0,0,0,0.09,0.93,0.97,0.85,0.36,0.28,1,1,0.38,0,0,0,0,0,0,0,0.49,1,1,1,1,1,1,1,1,0.7,0.51,0.45,0.48,0.57,0.24;0,0,0,0.01,0.04,0.16,0.51,0.79,0.26,0,0,0,0,0,0,0,0,0.3,0.65,0.64,0.46,0.52,0.61,1,1,0,0,0,0,0,0,0,0,0.93,1,1,1,1,1,1,0.27,0,0,0,0.03,0.15,0.43,0.55,0;0,0,0,0,0,0,0.12,0.48,0.21,0,0,0,0,0,0,0,0.16,0.39,0.48,0.61,0.66,0.83,1,0.81,0,0,0,0,0,0,0,0,0.65,1,0.82,0.59,1,1,0,0,0,0,0,0,0,0.32,0.87,0.59,0;0,0,0,0,0,0,0,0.25,0.29,0,0,0,0,0,0.16,0.46,0.82,0.8,0.55,0.48,0.52,0.28,0,0,0,0,0,0,0,0,0,0.25,1,1,0.94,1,0.9,0,0,0,0,0,0,0,0.6,1,0,0,0;0,0,0,0,0,0,0,0.04,0.31,0.29,0.02,0,0.17,0.59,0.86,1,0.88,0.48,0.08,0,0,0,0,0,0,0,0,0,0,0,0.89,1,1,1,1,0.58,0,0,0,0,0,0,0,1,0.46,0,0,0,0;0,0,0,0,0,0,0,0,0.09,0.47,0.66,0.61,1,1,1,1,0.53,0,0,0,0,0,0,0,0,0,0,0.41,0.91,1,1,1,1,1,0,0,0,0,0,0,0,0.61,0.86,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.13,0.83,1,1,1,1,0.77,0.29,0,0,0,0,0,0,0,0,0.91,1,1,1,1,1,1,0.84,0,0,0,0,0,0,0,0.76,0.1,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.73,1,1,1,0.91,0.53,0.24,0,0,0,0,0,0,0.78,1,1,1,1,1,1,1,0.55,0,0,0,0,0,0,0.01,0.94,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.08,0.82,1,1,0.96,0.62,0.42,0.27,0.12,0,0,0,0.19,1,0.86,0.93,0.9,1,1,1,1,0,0,0,0,0,0,0,0.41,0.76,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.25,0.82,0.93,0.79,0.61,0.49,0.4,0.34,0.29,0.26,0.46,1,1,0.97,0.85,1,1,1,0.96,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.22,0.61,0.7,0.47,0.23,0.16,0.17,0.21,0.37,0.74,1,1,0.94,0.93,1,1,0.92,0,0,0,0,0,0,0,0,0.73,0.75,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.02,0.26,0.5,0.51,0.32,0.17,0.14,0.22,0.5,1,1,0.71,0.42,1,1,0.62,0,0,0,0,0,0,0,0.74,0.95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.05,0.28,0.46,0.48,0.5,0.61,0.84,1,0.99,0.85,0.98,1,0.56,0,0,0,0,0,0,0.18,1,0.09,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.01,0.21,0.46,0.69,0.84,0.94,1,1,0.85,0.57,0.16,0,0,0,0,0,1,0.07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.23,0.67,0.84,0.73,0.52,0.58,0.36,0.13,0,0,0,0.01,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.46,0.69,0.4,0.18,0.45,0.44,0.41,0.5,0.89,1,0.15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3,0.52,0.31,0.13,0.09,0.17,0.44,0.79,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.17,0.37,0.36,0.31,0.38,0.51,0.19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.01,0.12,0.22,0.22,0.09,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        %case 'S+1'; name='Decadentium volubilis'; if isLoadCell; R=36;peaks=[2/3,1,2/3,1/3];mu=0.15;sigma=0.014;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.021647,0.258426,0.497772,0.301681,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.600861,0.966692,0.599068,0.20428,0,0,0,0,0,0,0.358747,0.576723,0.745393,0.889819,0.984285,0.736697,0.185045,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.411876,0.922204,1,1,0.494812,0.028091,0,0,0,0,0.369695,0.852807,0.983332,1,1,1,0.962888,0.453503,0.046557,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.165137,0.809842,1,1,1,0.864612,0.04867,0,0,0.196746,0.499782,0.721437,0.766152,1,1,1,1,0.929269,0.362247,0.04532,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.539539,1,1,1,1,1,0.435716,0.061417,0.369023,0.735874,0.81922,0.654574,0.222878,0.214402,0.780726,1,1,0.891554,0.155001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.813349,1,0.329073,0.993099,1,1,1,0.6308,0.747449,0.972995,0.762489,0.225104,0,0,0.498917,0.980153,1,0.892349,0.279993,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.265967,0.967033,0.806177,0.027169,0,0.868685,1,1,1,0.975085,1,0.449085,0.002903,0,0,0.526839,0.984437,1,0.974451,0.546498,0.374751,0.291549,0.23138,0.536095,0.388458,0.2122,0.095544,0.080022,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.532567,1,0.529596,0,0,0,0.998729,1,1,1,0.86717,0.172729,0,0,0,0.680679,1,1,1,0.806669,0.67572,0.617968,0.582204,0.950953,0.901479,0.750976,0.601357,0.54032,0.544065,0,0,0,0,0,0;0,0,0,0.225263,0.779865,0.004674,0,0,0,0,0.738502,1,0.398869,0,0,0,0.657395,1,1,1,0.557942,0.036091,0,0,0,0.851272,1,1,1,0.936289,0.73203,0.539386,0.389552,0.438092,1,1,0.984529,0.934397,0.918423,0.654274,0,0,0,0,0;0,0,0,0.675763,1,1,1,0.175988,0,0.169002,0.885877,1,0.427254,0,0,0,0,0.660648,0.930276,0.665241,0.242684,0,0,0,0,0.838122,1,1,1,0.831008,0.429727,0.101949,0,0,0.462092,1,1,1,1,1,0.351251,0,0,0,0;0,0,0,0.857396,1,1,1,1,0.957625,0.836615,0.990849,1,0.641682,0,0,0,0,0,0.116937,0.142757,0.028764,0,0,0,0,0.148826,1,1,0.894868,0.480422,0.068842,0,0,0,0.315364,0.870321,1,1,1,0.9038,0.504714,0,0,0,0;0,0,0,0.953294,1,1,1,1,1,1,1,1,0.774671,0,0,0,0,0,0,0,0,0,0,0,0,0,0.336222,0.538068,0.416354,0.133562,0,0,0,0.173224,0.544629,0.899836,1,1,0.83335,0.595728,0.337881,0.042653,0,0,0;0,0,0,0.984061,1,1,1,1,1,1,1,1,0.610697,0.011101,0,0,0,0,0,0,0,0,0,0,0,0,0,0.058677,0.050427,0,0,0.082027,0.290831,0.559191,0.837941,1,1,1,0.628319,0.243247,0.101179,0.001203,0,0,0;0,0,0,0.968933,1,1,0.138624,0.301657,0.841497,1,1,1,0.357139,0.003242,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.322688,0.717423,0.908482,1,1,1,0.946702,0.473729,0.150624,0,0,0,0,0;0,0,0,0.823665,1,1,0.733333,0,0,0.065198,0.3914,0.352249,0.075992,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.450753,0.988633,1,1,1,0.958901,0.751095,0.461228,0.315859,0,0,0,0,0;0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.355558,1,1,1,0.985746,0.810584,0.715132,0.587486,0.612552,0,0,0,0,0;0,0,0,0,0.988403,1,1,0.905705,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.166448,1,1,1,0.705393,0.589255,0.587312,0.644646,0.908628,0.415459,0,0,0,0;0,0,0,0,0,1,1,1,0.598035,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.008862,0.498893,0.683161,0.585341,0.225188,0.128866,0.15108,0.340646,1,0.806898,0.180766,0,0,0;0,0,0,0,0,1,1,1,0.88114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.07597,0.175847,0.077866,0,0,0,0,0.840067,1,0.605069,0.165361,0,0;0,0,0,0,0.418497,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.0083,0,0,0,0,0.445067,1,0.935336,0.568783,0.333128,0;0,0.33411,0.975498,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.287132,0.098822,0.02068,0.021268,0.115812,0.490995,1,1,0.901769,0.701438,0.290538;0.128741,1,1,1,1,1,1,1,0.065118,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.325838,0.658487,0.540715,0.469159,0.456413,0.517164,0.730733,1,1,1,0.969068,0.560597;0,1,1,1,1,1,1,0.297852,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.683185,0.958497,0.925214,0.886561,0.867674,0.882971,0.954905,1,1,1,1,0.683306;0,0.719586,1,1,1,0.346138,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.786799,1,1,1,1,1,1,1,1,1,1,0.592939;0,0.125727,1,1,0.995218,0.510909,0.273537,0.086957,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.546626,0.933607,1,1,1,1,1,0.975595,0.731597,0.8086,0.795507,0.364617;0,0,1,1,1,1,0.993846,0.904349,0.764995,0.526657,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.218526,0.463779,0.620581,0.794236,1,1,1,0.381711,0.390565,0.479577,0.436402,0.124679;0,0,0.217245,1,1,1,1,1,0.99714,0.818181,0.352618,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.006417,0.073631,0.190955,0.414661,0.752424,1,1,0.035254,0.087621,0.14304,0.113029,0;0,0,0,0.692255,1,1,1,1,1,0.958902,0.518495,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.057272,0.434934,1,0.913803,0,0,0,0,0;0,0,0,0.102338,0.715575,1,1,1,1,0.998361,0.572496,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.139285,0.096493,0,0,0,0,0.135919,1,0.863402,0,0,0,0,0;0,0,0,0,0.300416,0.567946,1,1,1,1,0.481517,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.171092,0.712932,0.482487,0.154799,0,0,0,0.013497,1,0.790471,0,0,0,0,0;0,0,0,0.043063,0.390027,0.690316,1,1,1,0.996578,0.220846,0,0,0,0,0.391258,0.06536,0,0,0,0,0,0,0,0,0,0,0,0,0.079028,0.757206,0.974806,0.82328,0.542997,0.214564,0,0,0.387091,1,0.727837,0,0,0,0,0;0,0,0.000118,0.3623,0.828899,1,1,1,1,0.959443,0,0,0,0,0.468889,0.871118,0.45533,0,0,0,0,0,0,0,0,0,0,0,0,0.276166,0.892674,1,1,0.867232,0.593297,0.331069,0.242784,0.84868,1,0.739364,0.114449,0,0,0,0;0,0,0.122556,0.744495,1,1,1,1,1,0.899145,0.01179,0,0,0.354407,1,0.978999,0.646717,0,0,0,0,0,0,0.085492,0.242917,0,0,0,0,0.166634,0.708186,1,1,1,0.894997,0.700065,0.65325,1,1,0.826345,0.343993,0,0,0,0;0,0,0.15228,0.724296,1,1,1,1,1,0.945424,0.671984,0.469424,0.928493,1,1,1,0.708667,0.024957,0,0,0,0.00631,0.408759,0.901026,0.616073,0.132009,0,0,0,0.024469,0.453481,1,1,1,1,0.944461,0.925767,1,1,0.93814,0.454998,0,0,0,0;0,0,0,0.132438,0.212209,0.159216,0.428222,1,1,1,1,1,1,1,1,1,0.704025,0,0,0,0,0.173151,0.771018,1,0.860207,0.379477,0,0,0,0,0.24368,1,1,1,1,1,1,1,1,1,0.058721,0,0,0,0;0,0,0,0,0,0,0,0.292555,0.896817,1,1,1,1,1,1,1,0.69522,0,0,0,0,0.237123,1,1,0.983847,0.59953,0,0,0,0,0.534948,1,1,0.906055,0.757367,0.959629,0.998219,1,1,1,0,0,0,0,0;0,0,0,0,0,0,0,0,0.130461,0.552013,1,1,1,1,1,1,0.7561,0.127526,0,0,0,0.928161,1,1,1,0.784499,0.201386,0,0,0,1,1,1,0.2163,0.473306,0.778701,0.920799,1,1,0.391355,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.061323,0.391266,0.877226,1,1,1,1,0.90636,0.448911,0.120173,0.151943,0.7931,1,1,1,1,0.924939,0.480153,0.071568,0.053208,0.875381,1,1,0.017205,0.029859,0.198433,0.500202,0.71562,0.786528,0.488842,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.139789,0.60579,1,1,1,1,1,0.893089,0.801037,0.957279,1,1,1,1,1,0.997595,0.796285,0.62907,0.922706,1,1,0.495951,0,0,0.0198,0.178496,0.30604,0.233289,0.009038,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.049469,0.47465,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.996339,1,1,1,0.851545,0,0,0,0,0,0.008877,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.008189,0.357634,1,1,1,1,1,1,1,0.945752,0.493184,0.441533,0.564776,0.956321,1,1,1,1,1,0.996447,0.322861,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.239461,0.815659,1,1,1,1,0.643361,0.000318,0.001307,0.012967,0.081803,0.31684,0.757289,1,1,1,1,1,0.812566,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.105733,0.513477,0.921239,1,0.750439,0,0,0,0,0,0,0.126746,0.522432,0.901782,1,1,1,1,0.079039,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.136776,0.289087,0.186359,0,0,0,0,0,0,0,0.002928,0.213243,0.543053,0.795585,0.953274,0.89037,0.366827,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.110715,0.25645,0.29969,0.178392,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case 'S+2'; name='Decadentium vagus'; if isLoadCell; R=36;peaks=[2/3,1,2/3,1/3];mu=0.13;sigma=0.012;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.005352,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.137535,0.324016,0.316909,0.207248,0.059269,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.283579,0.819745,0.801256,0.648271,0.498585,0.268722,0.033335,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.005162,0.005866,0,0,0,0,0,0,0.844574,1,1,0.918891,0.72621,0.614368,0.397235,0.083536,0,0,0,0,0,0,0,0.070525,0.090095,0.011304,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.170007,0.249537,0.116106,0,0,0,0,0.262869,0.932359,1,1,0.940912,0.753618,0.736431,0.674698,0.41354,0.093956,0,0,0,0,0.016552,0.440638,0.753425,0.772946,0.544305,0.234365,0.014408,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.272756,0.612592,0.485007,0.25894,0.03736,0,0,0.118335,0.979064,1,0.746064,0.137544,0.301695,0.660036,0.852819,0.724073,0.36369,0.104847,0.014509,0,0.208718,0.673856,0.993391,1,1,1,0.891543,0.458155,0.088531,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.146169,0.886088,0.766145,0.562983,0.362196,0.118553,0,0,1,1,0.053688,0,0,0.046484,0.792765,1,0.720066,0.345685,0.24157,0.313228,1,1,1,1,1,1,1,1,0.580877,0.131984,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.949163,1,0.762962,0.58965,0.435078,0.205614,0.058226,0.794253,1,0.413977,0,0,0,0.232303,0.829996,1,0.83375,0.73539,1,1,1,1,1,1,1,1,1,1,0.623827,0.122496,0,0,0,0,0,0,0,0,0;0,0.022954,0,0,0,0,0,0.252801,1,1,0.73501,0.606341,0.498126,0.297938,0.33378,1,1,0,0,0,0,0.370942,0.875838,1,1,1,1,1,0.798894,0.704284,0.820003,0.959687,1,1,1,1,0.580543,0.079538,0,0,0,0,0,0,0,0,0;0,0.001251,0,0,0,0,0,0.910952,1,0.987836,0.638177,0.583481,0.555367,0.411128,1,1,0.555003,0,0,0,0,0.447778,0.907856,1,1,1,0.683473,0,0,0.060503,0.454989,0.823127,1,1,1,1,0.491108,0.046695,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.370225,0.913035,1,0.582921,0.380067,0.490218,0.594384,1,1,1,0.317714,0,0,0,0,0.489819,0.890961,1,1,0.352937,0,0,0,0,0.279327,0.765362,1,1,1,1,0.523311,0.077749,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.36672,0.942256,0.946745,0.014365,0,0.308493,0.845853,1,1,1,0.297299,0,0,0,0,0.450266,0.803042,0.992114,0.579208,0.020021,0,0,0,0,0.301365,0.822327,1,1,1,1,0.963748,0.336217,0.00938,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.370036,0.993272,0.360564,0,0,0.128936,0.656818,0.931706,1,0.884993,0.22632,0,0,0,0,0.315164,0.627113,0.519872,0,0,0,0,0,0,0.461125,0.927386,1,1,1,1,1,1,0.372196,0.01114,0,0,0,0,0,0,0;0,0,0,0,0,0,0.445199,1,0.139237,0,0,0.042581,0.436061,0.765796,0.957069,0.55017,0.033509,0,0,0,0,0.08224,0.192931,0,0,0,0,0,0,0.06108,0.662726,0.988763,1,1,1,0.997126,0.985602,1,0.986636,0.378469,0.026464,0.070458,0.672907,0.715703,0.33977,0.010821,0;0,0,0,0,0,0,0.646673,1,0.102222,0,0,0,0.226358,0.546819,0.438476,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.318348,0.804439,1,1,0.485596,0,0,0.470635,0.914174,1,1,1,1,1,1,0.929702,0.229133,0;0,0,0.02482,0.041151,0.025951,0.007746,0.906677,1,0.202032,0,0,0,0.041467,0.305329,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.4303,0.82469,0.992046,0.429896,0,0,0,0.060944,0.835208,0.998692,1,1,1,1,1,1,0.362502,0;0,0.061526,0.228295,0.244405,0.191527,0.150275,1,1,0.448661,0.014978,0,0,0,0.065143,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.31396,0.710556,0.297205,0,0,0,0,0.061433,0.835713,1,1,1,1,1,1,0.931766,0.266687,0;0,0.308331,0.45168,0.424391,0.366129,0.589266,1,0.970305,0.710441,0.256828,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.226128,0.86697,1,0.765313,0.580998,1,1,1,0.491545,0.065636,0;0.001857,0.601342,0.643147,0.566021,0.513231,1,1,1,0.913158,0.566755,0.161519,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.390237,0.918166,1,0.875602,0.769594,1,1,0.699272,0.098916,0,0;0.058179,0.903544,0.90475,0.767946,0.677273,1,1,1,1,0.755235,0.34616,0.031002,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.576791,0.969872,1,1,1,1,1,0.187121,0,0,0;0.154812,1,1,1,1,1,0.999473,0.995531,0.992107,0.683854,0.278274,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.166506,0.738088,0.987778,1,1,1,1,1,0.000306,0,0,0;0.279873,0.98389,1,1,1,0.982059,0.869527,0.818992,0.775223,0.318156,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.441974,0.791181,0.938495,1,1,1,1,1,0,0,0,0;0.380337,0.787547,0.953388,1,1,0.918375,0.621393,0.516451,0.396708,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.402822,0.683528,0.56305,0.318479,0.234172,0.602589,1,1,0.151056,0,0,0;0.276402,0.514216,0.745594,0.986941,1,0.86619,0.335312,0.20024,0.139099,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.423245,1,0.692576,0,0,0;0.023006,0.247263,0.496736,0.871155,1,0.917992,0.113155,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.351897,1,1,0,0,0;0,0,0.248705,0.705548,1,1,0.097069,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.935792,1,1,0.018885,0,0;0,0,0,0.539354,1,1,0.379552,0.09602,0.061404,0.116694,0.245447,0.275876,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.004423,0.99284,1,1,1,1,0.698035,0,0;0,0,0,0.37811,0.985612,1,0.647921,0.552267,0.562681,0.82635,0.816728,0.740896,0.165021,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.082605,1,1,1,1,0.518434,0.370897,0.486084,0.632927,0.112893;0,0,0,0.254173,0.855513,0.832461,0.829921,0.850957,0.925241,1,1,0.95788,0.16109,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.968062,1,1,1,0.072286,0.086651,0.17543,0.445363,0.979083,0.759853;0,0,0,0.186301,0.545384,0.796443,0.896826,0.993633,1,1,1,0.790941,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.992183,0,0,0,0.051459,0.261656,0.664871,0.581179;0,0,0.036708,0.17161,0.568108,0.859275,1,1,1,0.978763,0.841101,0.282826,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.955556,1,0.505914,0,0,0,0,0.077712,0.254137,0.182052;0,0,0.104292,0.310892,0.89407,0.948113,1,1,0.981963,0.822656,0.562521,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.427757,1,1,0,0,0,0,0,0.017874,0;0,0,0.126271,0.423685,1,1,1,1,0.918335,0.609204,0.256044,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.999392,1,0.793847,0,0,0,0,0,0;0,0,0.065135,0.477652,1,1,1,1,0.94325,0.452776,0,0,0,0,0.070628,0.359827,0.401901,0.07547,0,0,0,0,0,0,0,0,0,0,0,0,0,0.104114,0.657465,0.880319,0.793225,0,0,0,0,1,0.994597,0,0,0,0,0,0;0,0,0,0.393211,0.733769,1,1,1,1,0.521518,0,0,0,0.05774,0.511223,0.884588,1,0.542232,0,0,0,0,0,0,0,0,0,0,0,0,0,0.554816,1,1,1,0.897283,0,0,0,0.453345,1,0.250242,0,0,0,0,0;0,0,0,0.178503,0.522433,0.774081,0.898458,0.989252,1,1,0.091265,0.054492,0.15968,0.503459,0.901144,1,1,0.97586,0,0,0,0,0,0,0,0,0,0,0,0,0,0.782191,1,1,1,1,0.86251,0,0,0.477391,1,0.160662,0,0,0,0,0;0,0,0,0,0.294103,0.311918,0.558516,0.686947,0.920475,1,0.734094,0.496759,0.616801,0.812515,0.96483,1,1,0.88977,0,0,0,0,0,0.154519,0.581444,0.499025,0,0,0,0,0,0.69429,1,1,1,1,0.996848,0.937194,0.999113,1,1,0,0,0,0,0,0;0,0,0,0,0.064042,0.109881,0.224829,0.304048,0.548987,0.837402,0.787021,0.630183,0.711803,0.788207,0.873491,1,1,0.747725,0,0,0,0,0.068438,0.744392,1,1,0.701112,0,0,0,0,0.358349,1,0.944148,0.414018,1,1,1,1,1,0.947016,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.165977,0.470233,0.503976,0.47755,0.630332,0.722922,0.80924,1,0.940123,0.59493,0,0,0,0,0.377278,1,1,1,0.976294,0,0,0,0,0.245749,1,0.644852,0,0.067842,0.386586,1,1,1,0.537407,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.099718,0.244394,0.331416,0.595521,0.739942,0.855175,1,0.892897,0.487705,0,0,0,0.065557,0.688938,0.940735,0.635298,1,0.988463,0.598983,0,0,0,0.391074,1,0.351866,0,0,0.077443,0.28571,0.423024,0.10109,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.015161,0.15227,0.572981,0.820538,1,1,0.942758,0.547578,0,0,0.018744,0.384921,0.835273,0.713871,0.244649,0.749386,1,0.693769,0,0,0.008087,0.682903,0.849636,0.019484,0,0,0,0.048508,0.06243,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.011598,0.190697,0.506661,0.913549,1,1,1,1,0.319763,0.15057,0.407818,0.66366,0.754329,0.437145,0.199628,0.400789,1,0.843312,0.282832,0,0.306825,0.772169,0.518198,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.148652,0.489401,0.919975,1,1,1,1,1,0.583635,0.54795,0.502223,0.38483,0.114636,0.136967,0.315561,0.948,0.990753,1,0.81157,0.744415,0.515241,0.033934,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.238363,0.667349,0.935203,1,0.909772,0.830172,0.798837,0.743565,0.169774,0.027409,0,0,0.143107,0.410389,0.715292,1,1,1,0.88035,0.411147,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.016449,0.353159,0.426311,0.452389,0.354103,0.296838,0.259425,0,0,0,0.004447,0.184185,0.475015,0.799696,1,1,0.874565,0.428809,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.137903,0.39806,0.677459,0.83645,0.821675,0.420019,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.012387,0.074297,0.082658,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end

        case 'S+3'; name='Astrium nausia'; if isLoadCell; R=18;peaks=[1,1/6,1/6,3/4];mu=0.22;sigma=0.021;dt=0.1;cells=[0,0,0,0,0,0,0,0,0.30077,1,1,0.624453,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.237882,1,1,1,1,0.006963,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.577876,0.969394,1,1,1,0.579864,0.142756,0,0,0,0,0,0,0,0,0.697407,0.894688,1,0.785098,0.47199,0,0,0,0;0,0,0,0,0.424139,0.8708,1,0.437627,0.172995,1,1,1,1,0.110634,0,0,0,0,0,0,0.499736,0.922011,0.997653,1,1,1,0.258056,0,0,0;0,0,0,0,0.896678,1,1,0.02841,0,0,0,0.059832,1,1,1,0.097206,0.261977,0.843823,1,1,1,0.495601,0.764515,1,1,1,0.930742,0,0,0;0,0,0,0,0.816056,1,1,0.746995,0,0,0,0,0.087379,1,1,1,0.927528,0.888214,0.548148,0,0,0,0,1,1,1,1,0,0,0;0,0,0,0,0,0,0.855148,1,0,0,0,1,1,1,1,0,0,0.312093,1,0.053539,0,0,0,0.246109,1,1,0.726153,0,0,0;0,0,0,0,0,0,0.539009,1,0.010482,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,1,1,0.012884,0,0,0;0,0,0,0,0,0,0.409434,0.998638,0.15739,1,1,1,0,0,0,0,0,0,1,1,0.997934,1,0,0.259257,1,1,0,0,0,0;0,0,0,0,0,0,0.483795,0.616302,0,0,0.985993,0,0,0,0,0,0,0,0,0.881781,1,1,0.524608,0.725511,1,0.536489,0,0,0,0;0,0,0,0,0,0,0.553078,0.408973,0,0,0,0,0,0.248332,0.194335,0.762208,1,0,0,0,0.517471,0.98914,0.730965,0.983414,1,0.362807,0,0,0,0;0,0,0,0,0,0.263192,0.665134,0.210239,0,0,0,0,1,1,0.854308,0.552056,1,1,0.223929,0,0,0.286334,0.533679,1,1,0.341168,0,0,0,0;0,0,0.077584,0,0.200097,0.834135,0.280309,0.029386,0.822026,0.238756,0,0,1,1,0.156278,0,0.743175,1,0.417581,0,0,0,0.639263,1,1,0.496941,0,0,0,0;0,0.775879,0.454347,0,0.720515,0.404537,0,0,1,0.876819,0,0,1,1,0.928064,0,0,1,0,0,0,0.45431,1,1,1,0.735555,0,0,0,0;0.674105,1,1,0.374341,1,0,0,0.473383,1,1,0,0,1,1,1,0.246256,0.761971,1,0,0,0,1,1,1,1,1,0.024724,0,0,0;0.84029,0.359604,0.361316,1,1,0,0,0.988612,0.227253,1,0,0,0.712038,1,1,0.783595,1,1,1,0,0.471932,1,1,0.158999,1,1,0.00401,0.141382,0,0;0.558732,0.194739,0,1,1,0,0,1,1,1,0.018175,0,0,0.661451,0.298641,0,0.987721,1,1,0.15132,1,0,1,0,0,1,0,1,0.691591,0;0.133481,1,0,0.606289,0.72804,0,0,0,1,0.253582,0,0.231128,0.373667,0,0,0,0,0.155636,0.359949,1,1,0,1,0,0,1,1,1,1,0.303405;0.342353,1,0.85299,0.745475,0.97898,0.499009,0,0,0,0,0,1,1,0,0,0,0,0.059108,0,0.211726,1,1,0.091484,0,0,1,1,1,1,0.483884;0.711188,0.997215,1,0.946418,1,1,1,1,0,0,0,0.705373,1,1,1,1,1,0.381995,0,0,0,0,0,0,0.955844,1,0.95342,0.268946,1,0.392736;0.730185,1,1,1,1,1,0.644822,0.837385,1,0,0,0.238961,1,1,1,1,1,0.067347,0,0,0,0.990658,1,1,1,1,0.994031,0.653788,1,0.011492;0,0.11749,0.495324,1,1,1,0,0,0.224805,1,0,0,0.583084,1,1,1,0.72049,0,0,0.975183,1,1,1,0.222195,0.776338,1,1,1,0.977563,0;0,0,0,0,0,0,0,0,0,0.756726,0.206129,0,0,0,0,0,0,0,0.999203,1,0.918644,0.808583,0.241782,0.018607,0,0.823806,1,1,0,0;0,0,0,0,0,0,0,0,0,0,1,0.138069,0,0,0,0,0,0.804448,1,1,0.990257,0.533334,0,0,0,0,0.295218,0,0,0;0,0,0,0,0,0,0,0,0,0,0.427351,1,1,1,1,1,1,1,1,1,0.80869,0.06797,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.552121,1,0.426972,1,1,1,1,1,1,0.833166,0.315058,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.684062,0.900506,0.053022,0,0,0.72751,1,1,0.747766,0.240973,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.210773,0.948683,1,0.295375,0.107327,0.639108,1,0.735687,0.120524,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.909954,1,1,1,1,0.228878,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.023956,0.446959,0.545575,0.301945,0.09633,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.04337,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case 'S+4'; name='Dodecadentium nausia'; if isLoadCell; R=14;peaks=[2/3,1,1/3];mu=0.27;sigma=0.033;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.401264,0.555894,0.178334,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.817437,0.809647,0,0,0,0,0,0.783059,0.578155,0,0,0,0,0,1,0.082349,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0.636648,1,1,0.992127,0,0,0.333008,1,1,1,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,1,1,1,0.105434,0,0,1,0.154061,1,1,0,0,1,1,1,1,0.416661,0,0,0,0,0;0,0,0,0,1,1,0,0,0,0,1,0.703998,0.222531,1,0,0,1,0,0.227716,1,0,0,1,1,1,1,0,0,0,0,0,0;0,0,0,0.661054,1,1,1,0,0,0,0,1,0.509106,1,0.786238,0.419989,1,0.344621,0.582357,1,0.442244,1,1,0.975055,1,1,0.134951,0,0,0,0,0;0,0,0,0,1,1,1,0.759662,0,0,0,1,1,0.397959,1,0.994135,1,0.903192,0.532024,1,1,0.830396,0,0.185729,1,1,0.212419,0,0,0,0,0;0,0,0,0,0.321741,1,1,1,1,0.883775,0.493312,0.778698,1,0.215015,0.203288,0.820932,0.903475,0.364175,0,0,0.106536,0.021422,0.012366,0.976544,1,1,0.956563,0.961534,0.885847,0.60161,0.054183,0;0,0,0,0,0,1,1,0.934874,0.091934,0.284751,1,1,1,0.348948,0,0,0.114016,0,0,0,0,0,0.837872,1,1,1,1,1,1,1,0.525954,0;0,0,0,0,0,0.413327,1,1,0.443764,0,0,0.424511,0.8738,0,0,0,0,0,0,0,0,0.268111,0.629685,0.983024,1,1,1,1,1,1,1,0.001795;0.432597,1,1,0,0,0,0,1,1,0.471994,0.011731,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.351623,0.843939,1,1,1,1,0.16892;1,1,1,1,1,1,1,1,1,0.98213,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.974179,1,1,1,1,0.068897,0.031533;1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0.805916,0,0,0;0,1,1,1,0.992284,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.550789,1,1,1,1,0.779835,0,0,0;0,0,0.655749,1,1,1,0.615914,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.149966,0.863028,1,1,1,0.120947,0,0;0,0,0,0.621308,1,1,1,1,0.450756,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.689762,1,1,1,0,0;0,0,0,0.444792,1,1,1,0.233029,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.016287,0.025282,0.924368,1,1,1,1,1,0;0,0.25013,1,1,1,0.852118,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0;1,1,1,1,1,0.585153,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.894025,1,0.031903,0,0,0.986555,1,1,1,0;1,1,1,1,1,1,1,1,0.924476,0.584889,0,0,0,0,0.107397,0,0,0,0.373197,0.976727,0.227904,0,0.094109,1,1,0.833915,0.060423,0,0,0,0,0;0.437138,1,1,1,0.946469,0.649463,0.995766,1,1,0.654485,0,0,0,0,0.487407,0.246745,0,0,1,1,1,1,0,0.040378,1,1,0.103537,0,0,0,0,0;0,0,0,0,0,0.33116,1,1,0.320558,0.003518,0.109819,0,0,0,1,1,0.31497,0,1,1,0.401669,1,1,1,1,1,0.098851,0,0,0,0,0;0,0,0,0,0,0.813783,1,0.364761,0,1,1,1,0.809145,0.442983,1,1,1,1,0.985354,1,0,0,0,1,1,1,0.35511,0,0,0,0,0;0,0,0,0,0,0.891504,1,0.902337,1,1,0.341907,1,1,0.442613,1,0.540304,0.529752,1,0.22373,1,0,0,0,0,1,1,1,0,0,0,0,0;0,0,0,0,0,1,1,1,1,0,0,0.900077,1,0,1,0,0,1,0.562946,1,0.75084,0,0,0,0.042388,1,1,0,0,0,0,0;0,0,0,0,0.799852,1,1,1,0,0,0,1,1,0.301068,1,0,0,0,1,1,0.866334,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,1,1,0.349602,0,0,0,1,1,1,0.426127,0,0,0,1,1,0.981154,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.061209,0,0,0,0,0.152305,1,1,0,0,0,0,0.2454,1,1,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0.725703,1,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case 'S+5'; name='Heptafolium nausia'; if isLoadCell; R=36;peaks=[1,1/2,1/3,5/12];mu=0.16;sigma=0.012;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.384984,0.656486,0.557705,0.272192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.079058,0.241067,0.248957,0.125074,0.063355,0.189478,1,1,0.894329,0.261253,0.218942,0.737932,1,1,0.879099,0.344645,0.30607,0.556558,0.726397,0.64079,0.35038,0.022679,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.009521,0.209992,0.547699,0.70735,0.619328,0.459694,0.615054,1,1,0,0,0,0,0,0.294275,1,1,1,1,1,1,1,1,1,0.777698,0.442699,0.122335,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.073805,0.348499,0.739917,0.97775,0.97101,0.736979,0.554252,1,1,0.832599,0,0,0,0,0,0.027979,1,1,1,1,1,1,1,1,1,1,1,0.607914,0.027008,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.110861,0.432364,0.792751,1,1,1,0.696353,0.672876,1,1,0.984868,0,0,0,0,0,0.84001,1,1,1,1,1,1,1,1,1,1,1,0.680848,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.060942,0.379298,0.690165,0.922985,1,1,1,1,1,1,1,1,0.852266,0,0,0,0.955598,1,1,1,1,0.95326,0.556321,0.418221,0.621661,1,1,1,0.999216,0.618586,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.054488,0.356959,0.569698,0.695969,0.842883,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.918109,0.101734,0,0,0.188635,0.856129,1,1,1,1,0.856712,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.184958,0.699331,0.842511,0.689806,0.654064,1,1,0.905879,0.728876,0.583255,0.550446,0.811943,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0.578836,1,1,1,1,1,1,0.069548,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.435896,1,1,1,1,1,1,0.953225,0.557338,0,0,0,0,0,0.749735,1,1,1,1,1,1,1,0,0,0,0,0,0,0.27771,1,1,1,1,0.712499,0.520024,1,0.535721,0,0,0,0,0,0,0;0,0,0,0,0,0.518019,0.999025,0.741688,0.911103,1,1,1,1,0.842951,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0.243347,0,0,0,1,0.761217,0,0,0,0,0,0;0,0,0,0,0.167276,0.976919,0,0,0,1,1,1,1,0.775658,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0;0,0,0,0,1,0.103403,0,0,0,1,1,1,1,0.70443,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.038703,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0;0,0,0,0.143333,0.975778,0,0,0,0,1,1,1,1,0.180837,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,0.628594,0,0,0,0;0,0,0,0.725893,0.904309,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0.700708,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0.091661,1,0.977778,0,0,0,0;0,0,0,1,1,0.306353,0,0.471064,1,1,1,1,0.13443,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0.978877,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0.88977,0,0,0,0;0,0,0.001252,0.868655,1,1,1,1,1,1,1,0.75793,0,0,0,0,0,0.717307,1,1,1,0.863306,0,0,0,0,0,1,1,1,0.87879,0,0,0,0,0,0,0.756068,1,1,1,1,1,1,1,0.811729,0,0,0,0;0,0,0.034737,0.483204,1,1,1,1,1,1,0.847141,0,0,0,0,0,0,0,1,1,1,0.943747,0,0,0,0,0,1,1,1,0.59846,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0.982053,0.606135,0.08054,0,0;0,0.020405,0.084666,0.204645,0.638011,1,1,1,1,0.793876,0,0,0,0,0,0,0,0,0.853266,1,1,0.314352,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0.942968,0.608867,0.032769,0;0,0.116495,0.168706,0.172228,1,1,1,0.971703,0.585997,0.024853,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.069026,0.502261,0.058829,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.004273,1,1,0.781166,0.359395,0;0.008942,0.198006,0.261158,1,1,1,0.946131,0.184759,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.218487,1,0.874652,0.405632,0;0.011551,0.191171,0.737611,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.017764,1,0.876595,0.412223,0.198526;0.005298,0.116159,0.968649,1,1,0.519644,0,0,0,0,0,0,0.734537,1,0.677329,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.141225,0.357599,0.398184,0.597937,0.142606,0,0,0,0,0,0,0,0,0,0.565643,1,0.761234,0.332765,0.23517;0,0.047043,0.657615,1,1,0,0,0,0,0,0,0.271717,1,1,1,1,0,0,0,0,0,0,0.983832,1,0.939158,0.51408,0,0,0,0,0.005294,0.928739,1,1,1,1,0.945715,0,0,0,0,0,0,0,0,1,1,0.530572,0.188837,0.182472;0,0.014329,0.238339,1,1,0.526325,0,0,0,0,0,0.749576,1,1,1,1,0,0,0,0,0,1,1,1,1,0.873271,0.059554,0,0,0,0,1,1,1,1,0.986175,0.201202,0,0,0,0,0,0.003202,0.493432,1,1,0.94006,0.406286,0.102846,0;0,0.010393,0.009088,0.985881,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,0.860441,0.089126,1,1,0,0,0,0,0,0.491925,1,1,0.740089,0,0,0,0,0,0,0.922385,1,1,1,1,0.952336,0.605953,0.149542,0;0,0.044443,0.35993,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.785137,0.513207,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0.958622,0.606378,0;0.00709,0.254775,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0.340605,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.007194,1,1,1,1,0.534156,0.532952,1,0.910187,0;0.074289,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.090787,0.741873,0.78907,0.284408,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.104547,1,1,1,0.007762,0,0,0.26083,1,0;0.236974,1,0.602805,0.519668,1,1,1,1,0.940454,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000815,0,0,0,0,0,0,0,0.522416,0,0,0,0,0,0,0,0,0.318442,1,1,1,0,0,0,0,1,0;0.359436,0.955598,0,0,0.183299,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0;0.242449,0.907553,0,0,0,1,1,1,1,0,0,0,0,0,0,0.002656,0.722473,0.480065,0,0,0,0,0,0,0,0,0,0,0,0.816489,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0.086631,0.705222,0;0.013557,1,0,0,0,1,1,0.906028,1,0.684038,0,0,0,0,0.329909,1,1,1,0.341476,0,0,0,0,0,0,0,0,0,0,0.341012,1,1,1,1,0,0,0,0,0,0.334199,1,1,1,1,0,0,0,1,0,0;0,1,0,0,0,1,1,0.609453,1,0.969788,0,0,0,0,0.826999,1,1,1,0.154251,0,0,0,0,0,0.547346,0.660372,0,0,0,0,0,1,1,1,0.984327,0,0,0,0,0.649831,1,1,1,1,0.354528,0.041432,0.709788,0.948933,0,0;0,0.402138,1,0.94233,1,1,1,0.980746,1,0.934752,0,0,0,0,0.887752,1,1,0.380515,0,0,0,0,0,0.63309,0.991788,0.952175,0.142622,0,0,0,0,0,0,0,0.147726,0,0,0,0,0,1,1,1,1,1,1,1,0.263791,0,0;0,0,1,1,1,1,1,1,1,0.756541,0,0,0,0,0.095499,0.692588,0.131018,0,0,0,0,0,0,0.997037,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.947893,1,1,1,0.788766,0.306208,0,0,0;0,0,0,0.158671,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.084057,1,1,0.597963,0,0,0,0,0;0,0,0,0,0.0043,0.09059,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.068118,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.086853,1,0.991474,0.213666,0,0,0,0,0;0,0,0,0,0,0,1,1,1,0.026486,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.264254,1,0.907836,0.041556,0,0,0,0,0;0,0,0,0,0,0,0.852293,1,1,1,0,0,0,0,0.794461,0,0,0,0,0,0,0,0,0,0,0.012,0,0,0,0,0,0,0,0.021002,1,1,1,0.307719,0.028705,0.032447,0.229119,0.709891,1,0.577803,0,0,0,0,0,0;0,0,0,0,0,0,0.026029,0.994839,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.037214,0.839852,1,0.999279,0.965732,1,1,1,0.80873,0.909351,1,0.702136,0.026595,0,0,0,0,0,0;0,0,0,0,0,0,0,0.546689,0.828419,0.989973,1,1,1,1,1,1,1,1,1,0.18624,0,0,0,0,0,0,0,0,0.154955,0.621307,1,1,1,0.87414,0.72346,0.744573,0.888286,0.999373,1,0.997747,0.828612,0.413948,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.375271,0.825618,1,1,1,1,1,1,1,1,1,1,0.26758,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0.994195,0.72687,0.296674,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.222981,1,1,1,1,1,1,1,0.578506,0.543143,1,1,0,0,0,0,0,0.015041,1,1,1,1,1,1,0.278633,0.611894,1,1,0.496836,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,1,1,0.385074,0.009045,0.010414,0.388212,1,1,0.434657,0.780634,1,0.735817,0,0,0,0,0.212121,1,1,1,1,0,0,0,0,0.304718,1,0.493934,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0.432699,1,1,0.792312,1,1,0.111954,0,0,0.078917,0.662927,1,1,1,0.993742,0,0,0,0,0.083674,1,0.345092,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,1,0.287926,0,0,0,0.007305,1,1,0.833256,1,1,0.950757,0.504703,0.514861,0.781697,1,1,1,1,0.195,0,0,0,0,0.777306,0.953586,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.531086,0,0,0.633004,1,1,0.283777,0.587153,1,1,1,1,1,1,0.95108,1,1,1,0.201,0.013448,0.210409,1,0.856169,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.889591,1,1,1,1,0,0,0,0,0.659262,0.908176,0.927207,0.791218,0.413142,0,0,0.343622,0.940714,1,1,0.749887,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case 'S+6'; name='Anulogeminium ventilans'; if isLoadCell; R=18;peaks=[1,1,1];mu=0.3;sigma=0.037;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.04888,0.315072,0.096361,0.000205,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.387739,0.759926,0.925484,0.728434,0.020873,0,0,0,0.176765,0.201149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.351335,0.281949,0.084881,0.034767,0.002561,0,0.602104,1,1,1,1,1,1,0.990056,0.965672,1,0.905872,0.595168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.12028,0.832975,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.884111,0.222008,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.115465,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.591011,0.275673,0.047863,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.000175,0.219861,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.804096,0.363364,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.003655,0.069272,0.905359,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.311311,0,0,0,0,0,0,0,0;0,0,0,0,0.014029,0.424383,0.60366,0.687361,0.893903,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.867194,0.013742,0,0,0,0,0,0,0;0,0,0,0,0.407839,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.545921,0,0,0,0,0,0,0;0,0,0,0,0.256346,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.742184,0.091723,0,0.082317,0.404904,0.594325,0.740996,1,1,1,1,1,1,1,1,1,1,1,0.415711,0,0,0,0,0,0;0,0,0,0,0.069587,0.997056,1,1,1,1,1,1,1,1,1,0.574651,0.09726,0.141911,0.064776,0,0,0,0,0,0,0,0.046729,0.615054,1,1,1,1,1,1,1,1,1,0.999033,0.425776,0.01047,0,0,0,0;0,0,0,0,0.046061,0.604149,1,1,1,1,1,1,1,1,1,0.021584,0,0,0,0,0,0,0,0,0,0,0,0,0.05985,0.666358,1,1,1,1,1,1,1,1,0.996752,0.375541,0.016035,0,0,0;0,0,0,0,0.067471,0.85853,1,1,1,1,0.992916,1,1,1,0.656601,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.735125,1,1,1,1,1,1,1,1,0.771133,0.086411,0,0,0;0,0,0.002761,0.102591,0.875643,1,1,1,1,1,1,0.632036,0.24617,0.1487,0.010932,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.100668,1,1,1,1,1,1,1,1,1,0.29468,0.000064,0,0;0,0.273073,0.750002,1,1,1,1,1,1,1,1,0.416818,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.353258,1,1,1,1,1,1,1,1,0.46085,0.005255,0,0;0.120801,0.856943,1,1,1,1,1,1,1,1,1,0.528624,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.416157,1,1,1,1,1,1,1,0.605547,0.012597,0,0;0.042732,0.872457,1,1,1,1,0.996529,1,1,1,1,0.371633,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.020062,1,1,1,1,1,1,1,0.804583,0.056768,0,0;0,0.238832,1,1,1,1,0.966179,0.908207,0.971638,0.899635,0.296541,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0.323868,0.018563,0;0,0.024067,0.41754,1,1,1,1,1,1,0.322389,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.829481,1,1,1,1,1,1,1,0.640277,0.085664,0;0,0,0.038831,1,1,1,1,1,1,0.634375,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.426825,1,1,1,1,1,1,1,0.886992,0.149147,0;0,0.001525,0.30152,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.145236,1,1,1,1,1,1,1,0.94711,0.272636,0.003824;0,0.203428,1,1,1,1,1,1,1,0.796346,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.219771,1,1,1,1,1,1,1,0.829471,0.1993,0.000228;0.228615,0.984235,1,1,1,1,1,1,1,0.253794,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.498127,1,1,1,1,1,1,1,0.64603,0.089621,0;0.656908,1,1,1,1,0.983705,0.941204,0.995716,1,0.186417,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.666804,1,1,1,1,1,1,1,0.488389,0.062774,0;0.51867,1,1,1,1,1,1,1,1,0.804886,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.68706,1,1,1,1,1,1,1,0.358324,0.056037,0;0.002087,0.467302,1,1,1,1,1,1,1,1,0.285768,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.786793,1,1,1,1,1,1,1,0.307716,0.062973,0;0,0,0.162841,1,1,1,1,1,1,1,0.258591,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.330457,1,1,1,1,1,1,1,1,0.400672,0.067355,0;0,0,0,0.281774,1,1,1,1,1,1,0.244386,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.872442,1,1,1,1,1,1,1,1,0.435479,0.050538,0;0,0,0,0.281996,1,1,1,1,1,1,0.952481,0.163902,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.323962,1,1,1,1,1,1,1,1,0.911104,0.281477,0.014427,0;0,0,0,0.737356,1,1,1,1,1,1,1,1,0.482914,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.082384,0.853558,1,1,1,1,1,1,1,1,0.556795,0.098898,0,0;0,0,0.262477,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.119252,0.921232,1,1,1,1,1,1,1,1,0.684448,0.126854,0.008525,0,0;0,0,0.428926,1,1,1,1,1,1,1,1,1,1,0.332922,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.912684,1,1,1,1,1,1,1,1,0.933502,0.193364,0.047617,0,0,0;0,0,0,0.59607,1,1,1,1,1,1,1,1,1,1,0.960601,0.687733,0.184975,0,0,0,0,0,0,0,0,0,0,0.119267,0.732258,1,1,1,1,1,1,1,1,1,0.614184,0.151214,0.000979,0,0,0;0,0,0,0,0.033594,0.380188,0.754144,1,1,1,1,1,1,1,1,1,1,0.187788,0,0,0,0,0,0,0,0.609156,1,1,1,1,1,1,1,1,1,1,1,1,0.427544,0.073114,0,0,0,0;0,0,0,0,0,0,0,0.452532,1,1,1,1,1,1,1,1,1,1,0.63011,0.975779,1,1,0.615165,0.496712,1,1,1,1,1,1,1,1,1,1,1,1,1,0.804958,0.230489,0.005643,0,0,0,0;0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.720489,0.392145,0.05268,0,0,0,0,0;0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.868287,0.378066,0.092522,0.032382,0,0,0,0,0,0;0,0,0,0,0,0,0,0.074259,0.859621,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.852959,0.104529,0.042311,0.00017,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.517773,0.71269,0.829009,0.98795,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.210214,0.02779,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.183754,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.876648,0.503852,0.035552,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.321567,1,1,1,1,1,1,1,1,1,1,1,1,0.758534,0.349979,0.296377,0.350339,0.31989,0.036902,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.536246,0.79218,0.808751,0.640538,0.211847,0.010256,0.467514,0.997529,1,1,0.863036,0.266688,0.00341,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.107751,0.210781,0,0,0,0,0,0.019432,0.444136,0.496385,0.189396,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.002054,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case 'S+7'; name='Hexafolium incarceratus'; if isLoadCell; R=18;peaks=[1/2,0,1];mu=0.2;sigma=0.027;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.006182,0.009638,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.027154,0.293235,0.400569,0.339311,0.173618,0.022747,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.160995,0.496299,0.507621,0.348996,0.119235,0.003153,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.306557,0.657536,0.487714,0.130212,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.208959,0.588571,0.251101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.244368,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.057308,0.132778,0.165744,0.131831,0.001885,0,0,0,0,0,0,0,0,0.141664,0.371454,0.039392,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.055671,0.796711,0.999177,0.981904,0.96651,0.805791,0.169915,0,0,0,0,0,0,0,0,0.524894,0.873805,0.428518,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.016287,1,1,1,1,1,0.703169,0.020423,0,0,0,0,0,0,0,0,0.849427,0.99978,0.447859,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0.629221,0,0,0,0,0,0,0,0,0,0.825526,0.908539,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.005468,0.0639,0,0,0,0,0,0,0,0,0,0,1,1,0.919541,1,1,1,0.556766,0,0,0,0,0,0,0,0,0,0,0.357101,0.17755,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.135449,0.19512,0.27024,0.208536,0,0,0,0,0,0,0,0,0,0,1,0.843808,0.190089,0.580278,1,0.797868,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.147583,0.372013,0.358297,0.778902,0.972442,0.031688,0,0,0,0,0,0,0,0,0.447353,1,0.245966,0,0,1,0.242509,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.036343,0.370581,1,1,1,1,1,0.010121,0,0,0,0,0,0,0.06879,1,1,0.099252,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.200074,0.560655,1,1,1,1,1,1,0.654874,0.251901,0.004784,0,0.072789,0.508816,0.909249,1,1,0.474755,0.002964,0.673436,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.129837,0.327159,0.711323,1,1,0.553755,0.146409,0.919379,1,1,0.878157,0.850452,1,1,1,1,1,1,0.660638,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.002801,0.742092,1,0.978001,0,0.320609,1,1,1,1,1,1,1,1,1,0.991518,0.737429,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0.737806,0.69063,1,1,1,1,1,0.851536,0.753532,0.059072,0,0,0,0.406383,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0.004551,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0.904951,0,0,0,0,0,0,0,0,0,0.937738,1,1,0.130759,0,0,0,0,0,0.44405,0.941974,0.568971,0,0,0,0,0,0,0;0.005135,0,0,0,0,0,0,0,0,0,0,0,0,0.835157,1,1,1,0.998049,0.790071,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0.657853,0.317699,0.33689,1,1,1,0.998261,0.737313,0.145345,0,0,0,0,0,0;0.002381,0,0,0,0,0,0,0,0,0,0,0,0,0.639589,1,1,1,0.932149,0,0,0,0,0,0,0,0,0,0,0,0,0.940862,1,1,1,1,1,0.828316,0.444245,0.979742,1,0.77948,0.555931,0,0,0,0,0,0;0.007937,0,0,0,0,0,0,0,0,0,0,0,0,0.466352,0.988524,1,1,0,0,0,0,0,0.939104,0.920758,0.362654,0,0,0,0,0,0,0.849166,1,1,0.721781,0.132784,0,0,0.037804,1,0.82853,0.555786,0,0,0,0,0,0;0.021772,0,0,0,0,0,0,0,0,0,0,0,0,0.445411,0.975638,1,1,0,0,0,0.087044,1,1,0.914679,0.277491,0.139996,0,0,0,0,0,0,0.631351,0.846289,0.210517,0,0,0,0.067015,1,0.868331,0.468838,0,0,0,0,0,0;0.013956,0,0,0,0,0,0,0,0,0,0,0,0.179533,0.609105,0.993555,1,1,0,0,0,0.784746,1,0.924451,0,0,0,0,0,0,0,0,0,0,0.649966,0.381416,0,0,0.110959,1,1,0.883679,0.392791,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.107416,0.466827,0.803061,0.995004,1,0.412857,0,0,0.272518,0.857035,0.907735,0.299222,0,0,0,0,0,0,0,0,0,0,0.932538,1,1,1,1,0.978588,0.702683,0.620301,0.193472,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.097736,0.518271,0.35245,0.636479,0.9642,1,0.455099,0,0,0.347397,0.718225,0.724944,0.278258,0,0,0,0,0,0,0,0,0,0,1,1,0.439482,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.06132,0.057952,0.003637,0.462065,0.549894,0.544016,0.531906,0.61346,0.844906,1,0.574413,0,0,0.087089,0.666067,0.614452,0.507099,0.019353,0,0,0,0,0,0,0,0,0.530029,1,0.444907,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.0928,0.14965,0.815479,1,0.921976,0.634586,0.629551,0.698136,1,1,0.771411,0,0,0,0.403804,0.610931,0.513791,0.303936,0.141242,0.057204,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.050442,0.146071,1,1,1,0.503899,0.588598,1,1,1,1,0,0,0,0,0.214327,0.356579,0.295014,0.185947,0.057771,0,0,0,0,0,0,1,0.915209,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.137577,1,1,1,1,1,1,1,1,1,0.480376,0,0,0,0,0.004525,0.025003,0,0,0,0,0,0,0,0.523301,1,0.262107,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.139069,1,1,0.740053,0.027504,0.558957,1,1,1,1,1,0.425077,0,0,0,0,0,0,0,0,0,0,0,0,0.759397,1,0.038737,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.473098,0,0,0,0,0,0.378023,0.723785,1,1,1,0.282104,0,0,0,0,0,0,0,0,0,0.007274,0.136411,0.811117,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0.346028;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.533249,0.861854,1,1,1,0.125413,0.004343,0,0,0,0,0.017312,1,1,1,0.749658,0.365795,0.872829,1,0.121439,0,0,0,0,0,0,0,0,0,0,0,0.179558,0.314353;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.300518,0.710652,0.955607,1,1,1,0.501139,0.303258,0.143069,0.545624,1,1,1,1,1,0.832926,0,0,0.82412,1,0,0,0,0,0,0,0,0,0,0,0,0.298605,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.601844,0.889121,1,1,1,1,1,1,1,1,0.464581,0.512868,1,1,0.739086,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0.008476,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.068464,0.837316,0.995452,0.484666,0.500518,1,1,0.999582,0,0,0,0.134754,0.411163,1,1,0.017966,0,0.058929,1,1,0.756936,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.673415,0.575754,0.287775,0.252174,1,1,0,0,0,0,0,0,0.105894,1,0.791191,0.289815,0.427912,0.982726,0.700668,0.459988,0.160137,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.485322,0.56444,0.264041,0.131013,1,1,0,0,0,0,0,0,0,0,0.826136,0.946888,0.907637,0.649867,0.426743,0.29535,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.080541,0.374141,0.486239,0.632406,0.568978,1,0,0,0,0,0,0,0,0,0,0.018757,0.633144,0.584343,0.376092,0.289438,0.225292,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.030636,0.231288,0.646899,1,1,1,0,0,0,0,0,0,0,0,0,0,0.17542,0.166551,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.006042,0.129156,0.514246,0.97479,0.712952,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.012829,0.042535,0.288972,0.375616,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.078074,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end

        case 'S+8'; name='Hydrogeminium natans'; if isLoadCell; R=18;peaks=[1/2,1,2/3];mu=0.26;sigma=0.036;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.001907,0.00931,0.015437,0.019935,0.020704,0.015936,0.009917,0.000622,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.003214,0.041961,0.096654,0.149715,0.187035,0.199031,0.192306,0.181136,0.163194,0.140211,0.104019,0.06022,0.017502,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.041566,0.149625,0.26418,0.369761,0.456372,0.508278,0.517064,0.49133,0.449453,0.405175,0.364578,0.316282,0.248475,0.16239,0.073841,0.013442,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.032126,0.182142,0.337,0.493904,0.641818,0.760072,0.829313,0.840922,0.799817,0.731154,0.656167,0.592496,0.536171,0.469932,0.374262,0.253017,0.125297,0.033902,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.187395,0.391923,0.576777,0.747139,0.906953,1,1,1,1,1,0.896418,0.796481,0.723479,0.656279,0.569715,0.451758,0.299963,0.149811,0.044186,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.224751,0.556111,0.857713,1,1,1,1,1,1,1,1,1,1,0.905346,0.820017,0.731152,0.61811,0.469167,0.295389,0.141104,0.041292,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.29257,0.737364,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.889018,0.766477,0.618977,0.434334,0.250174,0.109843,0.025709,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.115656,0.645548,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.928636,0.767928,0.568199,0.35317,0.183365,0.069783,0.007651,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.075283,0.600017,1,1,1,1,1,1,1,1,1,1,0.991823,0.969945,0.991924,1,1,1,1,1,0.929872,0.719223,0.469294,0.256707,0.115682,0.027289,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.287785,0.959473,1,1,1,1,1,0.994814,0.861678,0.819534,0.889959,0.942566,0.937967,0.895396,0.865909,0.907129,0.987963,1,1,1,1,1,0.874445,0.605147,0.343972,0.165528,0.05065,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.414245,0.979213,1,1,1,1,1,1,0.900755,0.632633,0.487279,0.533919,0.659246,0.7345,0.749526,0.759859,0.818189,0.934056,1,1,1,1,1,1,0.745135,0.450067,0.223762,0.079542,0.004027;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.052116,0.991004,1,1,1,1,1,0.999217,0.897721,0.643776,0.331426,0.143268,0.148132,0.300002,0.46868,0.572554,0.634163,0.72198,0.869331,0.989009,1,1,1,1,1,0.859956,0.557641,0.292575,0.110913,0.011044;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.592207,0.971672,1,1,1,0.675723,0.263111,0.273463,0.303254,0.159581,0,0,0,0,0.167287,0.344221,0.456201,0.574235,0.758666,0.954713,1,1,1,1,1,0.93925,0.642596,0.358864,0.147623,0.018402;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.403979,0.721963,0.911179,1,1,1,0,0,0,0,0,0,0,0,0,0,0.107347,0.24474,0.373747,0.575823,0.864866,1,1,1,1,1,0.992813,0.694768,0.405433,0.17705,0.028214;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000043,0.413148,0.794423,0.964775,1,1,1,0.820049,0,0,0,0,0,0,0,0,0,0,0,0.080807,0.204646,0.389071,0.718369,0.975018,1,1,1,1,1,0.710511,0.423682,0.193307,0.032901;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.050011,0.613422,0.90278,1,1,1,1,1,0.064999,0,0,0,0,0,0,0,0,0,0,0,0,0.121114,0.295407,0.618715,0.928769,1,1,1,1,1,0.68877,0.410518,0.186266,0.028989;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.50635,0.86545,1,1,1,0.214717,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.125304,0.340338,0.670893,0.927159,1,1,1,1,0.969485,0.637083,0.370931,0.158524,0.015167;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.208159,0.588331,1,1,0.626125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.188706,0.510137,0.835101,0.979115,1,1,1,1,0.912713,0.564545,0.313538,0.118443,0.002723;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.249071,0.464265,0.634598,1,1,0.216379,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.258977,0.688406,0.959882,1,1,1,1,1,0.839739,0.477922,0.248396,0.073154,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.020289,0.628683,1,1,1,1,1,0.432034,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.268779,0.771743,1,1,1,1,1,1,0.735724,0.387709,0.176562,0.028889,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000203,0.010038,0.607164,0.979288,1,1,1,1,0.989763,0.818841,0.552449,0.148354,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.23966,0.789433,1,1,1,1,1,0.998956,0.587905,0.286143,0.103027,0.000685,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06596,0.060862,0.481546,1,1,1,0.887568,0.126065,0.089154,0.286189,0.436072,0.325828,0.029128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.340739,0.851277,1,1,1,1,1,0.744679,0.406854,0.169608,0.031018,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.018977,0.092363,0.238139,0.458597,1,1,1,0.615026,0,0,0,0.075764,0.154536,0.008341,0,0,0,0,0,0,0,0,0,0,0,0,0.190366,0.528786,0.779666,1,1,1,1,1,0.743372,0.48766,0.222795,0.049443,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.035603,0.051485,0.484757,1,1,1,1,1,1,0.094511,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.07213,0.944767,1,1,1,1,1,1,0.666447,0.423879,0.257909,0.059872,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.050843,0.201629,0.370932,0.753206,1,1,1,1,1,1,1,1,0.281489,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.422979,1,1,1,1,1,1,0.800904,0.31522,0.1555,0.050584,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.410542,0.307997,0.486687,0.990103,1,1,1,0.645802,0.813247,1,1,1,0.940951,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.962994,1,1,1,1,1,0.911063,0.264245,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0.299148,0.665091,0.625269,0.731106,1,1,1,0.708053,0,0,0.225523,1,1,0.851304,0,0,0,0,0,0,0,0,0,0,0.052747,0.048571,0,0,0.713735,1,1,1,1,1,0.818987,0.201071,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.129592,0.498125,1,1,1,1,1,1,1,0,0,0,0,0.20705,0,0,0,0,0,0.04654,0.111396,0,0,0,0,0.139203,0.968845,1,1,1,1,1,0.968997,0.857262,0.304363,0.057074,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.045659,0.436849,0.633484,1,1,1,1,1,1,1,1,1,0.14495,0,0,0,0,0,0,0,0,0.558555,0.697426,0.258488,0,0,0,0.260486,1,1,1,1,0.998301,0.895095,0.707086,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.73201,0.754377,0.942505,1,1,1,0.841802,0.607154,0.743357,0.994148,1,1,1,0,0,0,0,0,0,0,0.048527,0.8162,1,0.828995,0.184086,0,0.01305,0.613784,1,0.839865,0.730054,0.876214,0.833574,0.58752,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.506285,1,1,1,1,1,0.948954,0,0,0,0.671293,1,1,0.977738,0,0,0,0,0,0,0,0,0.498248,1,1,1,0.747459,0.743563,1,1,0.635718,0.433971,0.634851,0.6403,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.112444,0.887219,1,1,1,1,1,0.678529,0.02837,0,0,0,0,0.57429,0,0,0,0,0.028707,0.871785,0.794132,0.041143,0,0.008852,0.942593,1,1,1,1,1,0.948534,0.397472,0.100628,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0.450491,0.751272,0.933087,1,1,1,0.598849,0.54637,0.58078,0.403606,0,0,0,0,0,0,0,0,0,0.36265,1,1,0.528068,0,0,0.634383,1,1,0.948451,1,0.963501,0.450231,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.43898,0.99114,0.936634,0.975696,1,1,0.0732,0,0,0.561208,1,0.403276,0,0,0,0,0,0,0,0,0.499293,1,1,1,0.04283,0.015114,0.914592,1,0.748615,0.16685,0.542252,0.522776,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.75498,0.998421,0.995494,1,1,0.910854,0,0,0,0.274176,1,0.620871,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0.552686,0,0.012592,0.107157,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.863033,0.997433,1,1,1,0.406449,0,0,0,0,0.066981,0,0,0,0,0,0,0.575879,0,0,0,0.701732,1,1,1,1,1,0.986248,0.30322,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.67962,0.989301,1,1,1,0.659872,0.113175,0,0,0,0,0,0,0,0,0,0.628721,1,0.878381,0,0,0,1,1,1,0.908762,0.893801,0.651251,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0.019362,0,0,0.498842,0.978492,0.57906,0.339773,0.428738,1,1,0.109069,0,0,0,0,0,0,0,0,0.37698,0.989518,0.953546,0.113227,0,0.036483,1,1,0.45991,0.362463,0.675292,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0.378189,0.958048,0.610737,0.279009,0.474583,1,1,0.340866,0,0,0,0,0,0,0,0,0,0.353003,0.764979,0.914911,0.999723,1,1,1,0.002573,0,0.106138,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0.005348,0.345879,0.77184,0.808297,0.392884,0.607602,1,1,0.073389,0,0,0,0,0,0,0,0,0,0,0.644041,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.086828,0.183347,0.428747,0.854934,1,0.791405,0.70447,0.677087,0.354786,0.026263,0,0,0,0,0.795799,0.701337,0,0,0,0,0.805583,1,1,1,1,0.427174,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.149622,0.282336,0.608506,0.79771,1,1,0.715272,0.497946,0.282614,0.050418,0,0,0,0.136241,1,1,0.23813,0,0,0.407586,1,1,1,1,0.543427,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.667183,0.775213,0.747185,0.767882,0.927074,1,0.696822,0.51112,0.383324,0.197228,0.03949,0.005157,0.040381,0.230468,0.439924,0.690309,0.160045,0.014753,0.199429,1,1,1,0.677078,0.437427,0.003944,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.709083,0.837531,0.80686,0.740806,0.80722,0.770329,0.603343,0.518597,0.490873,0.455332,0.35226,0.288688,0.26713,0.16559,0.036147,0.005756,0.031685,0.978055,1,1,1,1,0.742703,0.438269,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.469228,0.72809,0.757568,0.709054,0.538084,0.474449,0.50627,0.529458,0.585743,0.764231,0.920161,0.842218,0.631921,0.26047,0.026671,0.00084,0.319598,1,1,1,0.973903,0.403168,0.289532,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0.066052,0.378157,0.506431,0.337827,0.294828,0.327704,0.446547,0.589177,0.777911,1,1,1,0.90134,0.615354,0.346159,0.302803,1,1,1,0.964313,0.53995,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.057182,0.219813,0.417769,0.718555,1,1,1,1,0.890972,0.812166,1,1,1,1,1,0.976591,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.028319,0.376548,0.864861,1,1,1,0.929752,0.707785,0.806247,0.783776,0.830185,0.896066,0.944421,0.973832,0.906588,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.303778,0.649488,0.974595,1,0.989527,0.745641,0.762108,0.583058,0.478291,0.408072,0.436682,0.469184,0.419116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.042562,0.278074,0.614799,0.873701,0.799142,0.575456,0.57355,0.430194,0.337208,0.218874,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.001761,0.027598,0.064294,0.275219,0.255243,0.236831,0.246352,0.265928,0.002641,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.079683,0.105754,0,0,0,0,0,0,0,0,0.101077,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0.014225,0.124424,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; end
        case 'S+9'; name='Heptapteryx liquefaciens'; if isLoadCell; R=10;peaks=[3/4,1,1];mu=0.34;sigma=0.051;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.009269,0.01899,0.025194,0.029591,0.033762,0.039955,0.04534,0.046165,0.042142,0.038301,0.037836,0.04176,0.049104,0.058804,0.068079,0.070892,0.063853,0.049799,0.028951,0.013836,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.007091,0.017213,0.03399,0.059556,0.092555,0.129334,0.16607,0.190983,0.207645,0.22516,0.242233,0.256761,0.265619,0.272004,0.273339,0.270747,0.270317,0.272388,0.273808,0.275241,0.276248,0.269087,0.250621,0.226446,0.207278,0.173323,0.126663,0.07145,0.026106,0.002769,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0.016394,0.037753,0.053407,0.073138,0.103967,0.139361,0.183101,0.229962,0.28079,0.334073,0.385517,0.427465,0.458108,0.486283,0.524314,0.554155,0.572874,0.58519,0.5922,0.590769,0.582339,0.575065,0.573653,0.570236,0.555353,0.528801,0.501771,0.4785,0.457038,0.429462,0.400327,0.357455,0.305488,0.238033,0.170558,0.105223,0.054435,0.020614,0.004033,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.036415,0.103856,0.162688,0.193093,0.224132,0.265993,0.320717,0.377516,0.443283,0.502086,0.564111,0.621327,0.680511,0.72973,0.775968,0.813108,0.854769,0.885846,0.904264,0.911856,0.903259,0.880477,0.853729,0.831912,0.817843,0.806714,0.799234,0.786432,0.759274,0.729862,0.702851,0.683923,0.665648,0.637545,0.594486,0.526885,0.442642,0.358112,0.283746,0.223592,0.170564,0.115038,0.067336,0.032735,0.008489,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.020909,0.117112,0.240039,0.3494,0.412702,0.4499,0.500303,0.553216,0.610456,0.669729,0.732265,0.782781,0.830748,0.86537,0.904863,0.95014,1,1,1,1,1,1,1,1,0.967783,0.944986,0.945054,0.948438,0.960296,0.978462,0.98035,0.963613,0.947538,0.932666,0.904883,0.863185,0.80992,0.763334,0.706092,0.644018,0.587872,0.532148,0.480928,0.40813,0.32336,0.247733,0.173425,0.100389,0.040562,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0.057796,0.206711,0.38643,0.540647,0.64941,0.711878,0.765815,0.818888,0.86236,0.914619,0.966355,1,1,0.982104,0.952173,0.956311,1,1,1,1,1,1,1,0.984003,0.923105,0.845835,0.817125,0.86473,0.947732,0.995754,1,1,1,1,1,1,0.998855,0.934193,0.879879,0.846614,0.829048,0.824172,0.802689,0.76194,0.706784,0.626019,0.534698,0.444342,0.347056,0.242593,0.140132,0.040688,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0.082494,0.278778,0.499245,0.690174,0.829714,0.934923,1,1,1,1,1,1,1,0.985022,0.893034,0.807866,0.800206,0.857008,0.912612,0.930528,0.946952,0.945989,0.919593,0.878992,0.772158,0.646021,0.573318,0.563996,0.615005,0.730208,0.866079,0.94371,0.987259,1,1,0.999117,0.991553,0.985757,0.91504,0.821986,0.794392,0.832602,0.907812,0.970748,0.968107,0.936145,0.890706,0.816652,0.724927,0.623749,0.506468,0.370479,0.214738,0.066713,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0.090081,0.326923,0.58315,0.801304,0.957371,1,1,1,1,1,1,1,0.986524,0.915833,0.79622,0.625082,0.538455,0.546285,0.602669,0.68441,0.761988,0.804016,0.786381,0.730817,0.648213,0.476531,0.320235,0.283486,0.310051,0.359431,0.46046,0.648288,0.808296,0.915724,0.981507,0.986863,0.93926,0.896323,0.847644,0.691702,0.608569,0.64305,0.707864,0.814225,0.958315,1,1,1,1,0.988592,0.88534,0.758274,0.606777,0.429929,0.237662,0.066835,0,0,0,0,0,0,0;0,0,0,0,0,0.080887,0.338099,0.634794,0.884561,1,1,1,1,1,1,1,0.990538,0.936181,0.841394,0.695313,0.484897,0.329168,0.277893,0.305634,0.40022,0.524681,0.622068,0.704991,0.680594,0.587594,0.494434,0.288862,0.100117,0.044755,0.055361,0.100736,0.214694,0.424959,0.63747,0.791739,0.919623,0.948821,0.857598,0.746914,0.637657,0.457456,0.380937,0.4122,0.469204,0.557238,0.757928,0.952884,1,1,1,1,1,1,0.852617,0.673495,0.462525,0.233075,0.052084,0,0,0,0,0,0;0,0,0,0,0.035153,0.287934,0.622398,0.915797,1,1,1,1,0.99455,0.965252,0.944138,0.919366,0.841657,0.738676,0.62734,0.435672,0.182344,0.06875,0.036502,0.067236,0.217059,0.372022,0.475746,0.591134,0.593328,0.459972,0.321919,0.127088,0,0,0,0,0,0.166879,0.391244,0.588455,0.768821,0.830941,0.712761,0.543681,0.418064,0.277205,0.171271,0.153934,0.219687,0.365357,0.598114,0.804537,0.931799,0.996932,1,1,1,1,1,0.917172,0.70724,0.441581,0.169254,0.011207,0,0,0,0,0;0,0,0,0,0.182451,0.533708,0.886862,1,1,1,1,0.942486,0.83926,0.762741,0.736575,0.707404,0.612871,0.510324,0.371116,0.11217,0,0,0,0,0.022247,0.222251,0.33666,0.499888,0.563386,0.396499,0.17043,0,0,0,0,0,0,0,0.137488,0.43706,0.638792,0.675608,0.52808,0.304753,0.159514,0.054178,0,0,0.065568,0.229422,0.425735,0.604793,0.772716,0.922362,0.989781,1,1,1,1,1,0.92529,0.639123,0.296893,0.043802,0,0,0,0,0;0,0,0,0.073219,0.388068,0.772586,1,1,1,1,0.891894,0.750233,0.623697,0.533699,0.488874,0.439446,0.350323,0.255647,0.076341,0,0,0,0,0,0,0.175525,0.357694,0.60066,0.723934,0.542498,0.210167,0,0,0,0,0,0,0,0.036194,0.485964,0.661956,0.645636,0.488653,0.188813,0,0,0,0,0,0.054429,0.203157,0.401338,0.623506,0.816647,0.926455,0.97234,1,1,1,1,1,0.861216,0.486557,0.134442,0,0,0,0,0;0,0,0.004705,0.19326,0.543827,0.951196,1,1,1,0.910347,0.691679,0.535213,0.430592,0.349761,0.276283,0.193627,0.111216,0.018871,0,0,0,0,0,0.016072,0.060383,0.290044,0.593018,0.870021,0.954479,0.808153,0.501783,0.065651,0.009588,0,0,0,0,0.018496,0.13953,0.626367,0.751931,0.720343,0.600841,0.254489,0,0,0,0,0,0,0.009796,0.28774,0.55799,0.727049,0.834265,0.900027,0.94683,0.983218,1,1,1,1,0.792802,0.368328,0.045808,0,0,0,0;0,0,0.029925,0.228257,0.613272,1,1,1,0.974903,0.755757,0.510643,0.358928,0.249339,0.151841,0.062688,0,0,0,0,0.001736,0.012999,0.030334,0.109059,0.232099,0.304706,0.480271,0.838063,1,1,0.92555,0.71927,0.348969,0.227351,0.168397,0.106045,0.11997,0.183769,0.252057,0.383297,0.719106,0.859334,0.858665,0.724951,0.382999,0.086781,0.026523,0,0,0,0,0,0.186572,0.387947,0.508948,0.646037,0.728869,0.749048,0.823616,0.950943,1,1,1,1,0.640129,0.180716,0,0,0,0;0,0,0.016929,0.217708,0.677848,1,1,1,0.890509,0.646366,0.420565,0.240013,0.08196,0,0,0,0.005278,0.085127,0.137039,0.185041,0.236351,0.281441,0.387275,0.485322,0.50384,0.551594,0.811136,0.906421,0.900127,0.79823,0.608766,0.459354,0.446464,0.429707,0.387027,0.384091,0.426027,0.45411,0.440307,0.603386,0.783337,0.853224,0.735293,0.487453,0.358836,0.271364,0.171454,0.059397,0.01845,0,0,0.0529,0.14295,0.22163,0.372133,0.447829,0.46588,0.580589,0.787885,0.948437,1,1,1,0.924798,0.349024,0.021046,0,0,0;0,0,0.009396,0.283773,0.860633,1,1,1,0.818379,0.547144,0.334361,0.107671,0,0,0,0,0.166657,0.340524,0.381348,0.410461,0.422875,0.432582,0.445845,0.404867,0.345118,0.29168,0.484919,0.606783,0.614588,0.485394,0.281749,0.220724,0.310859,0.335374,0.338814,0.34429,0.347099,0.297709,0.19173,0.307328,0.497791,0.601475,0.528857,0.409224,0.439917,0.459166,0.427296,0.331984,0.262268,0.177403,0.105745,0.131453,0.167527,0.126838,0.15308,0.141903,0.119536,0.210723,0.396873,0.605998,0.869263,1,1,1,0.656081,0.147902,0,0,0;0,0,0.034734,0.503747,1,1,1,0.992716,0.773808,0.438092,0.153867,0,0,0,0,0.152618,0.440984,0.498335,0.408245,0.329907,0.264213,0.216802,0.191466,0.106521,0.004863,0,0.107353,0.260241,0.270633,0.12742,0,0,0.007475,0.030372,0.041024,0.051215,0.066244,0.006705,0,0.037393,0.231569,0.334906,0.258289,0.164826,0.224949,0.329365,0.416528,0.446186,0.437817,0.410139,0.37382,0.414045,0.416587,0.112076,0,0,0,0,0.03373,0.289845,0.672615,0.983333,1,1,1,0.438656,0.045455,0,0;0,0,0.123824,0.692844,1,1,1,0.997821,0.763733,0.311378,0,0,0,0,0.132153,0.438874,0.572917,0.36842,0.175418,0.081361,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06064,0.15956,0.045965,0,0,0.073584,0.158519,0.218085,0.28796,0.367041,0.423782,0.482716,0.458467,0.207439,0,0,0,0,0,0.048893,0.450742,0.888289,1,1,1,0.739933,0.194458,0,0;0,0,0.193911,0.661371,1,1,1,1,0.686138,0.144852,0,0,0,0.100197,0.456465,0.645792,0.41994,0.068928,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.036777,0.13328,0.216018,0.364178,0.541073,0.393643,0.05514,0,0,0,0,0,0.023972,0.478631,0.935169,1,1,0.884835,0.239767,0,0;0,0,0.206814,0.613755,1,1,1,1,0.420238,0,0.001681,0,0.003623,0.365258,0.672308,0.543015,0.122883,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.108452,0.398903,0.552105,0.305534,0.014561,0,0,0,0,0,0.137846,0.689927,1,1,1,0.361954,0.0105,0;0,0,0.190213,0.811237,1,1,1,1,0.082124,0,0.069361,0.018156,0.110883,0.594446,0.624635,0.271317,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.169575,0.579429,0.507411,0.032456,0,0,0,0,0,0.126351,0.614305,1,1,1,0.72344,0.191977,0;0,0,0.168377,1,1,1,1,0.792808,0.013626,0.038803,0.142841,0.143746,0.391969,0.638692,0.394323,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.621537,0.748454,0.188148,0.006606,0,0,0,0,0.088769,0.466058,0.889261,1,1,0.965024,0.415871,0.033484;0,0,0.193166,1,1,1,0.936091,0.128316,0.003616,0.058503,0.156899,0.277857,0.595047,0.584313,0.154624,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.477056,0.987865,0.702824,0.267572,0,0,0,0,0.125542,0.558129,0.896803,1,1,1,0.406793,0.051215;0,0,0.37414,1,1,1,0.418469,0,0,0.02626,0.110501,0.379154,0.771336,0.708188,0.12903,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.211642,0.802212,0.952152,0.416296,0,0,0,0.093943,0.886405,1,1,1,1,1,0.300405,0;0,0,0.722098,1,1,0.934989,0.412118,0,0,0.002161,0.065619,0.543656,0.956538,0.849162,0.180954,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.406287,0.980545,0.642547,0.051301,0.055803,0.484836,0.243248,0.65056,0.969908,1,1,1,1,0.243721,0;0,0.104368,1,1,0.851104,0.80414,0.552544,0,0,0,0.001834,0.630769,1,0.856633,0.084385,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.006495,0.854858,1,0.580068,0.597175,0.270009,0,0.205574,0.937313,1,1,1,1,0.518651,0;0,0.144669,1,1,0.708062,0.654376,0.251624,0,0,0,0,0.456706,1,0.749952,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.489614,1,1,0.55216,0.223895,0,1,1,1,1,1,1,0.745815,0;0,0.261406,1,1,0.639576,0.423991,0.157011,0,0,0,0,0.466319,1,0.676237,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.873225,1,0.888399,1,0.780825,1,0.995945,1,1,1,1,0.528232,0.027655;0,0.704473,1,0.997096,0.487751,0.220605,0.061365,0,0,0,0,0.990594,1,0.57078,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.769131,1,1,1,0.27362,0,0.125418,0.816805,0.919669,1,1,0.55312,0.009978;0,1,1,0.998151,0.404786,0.219614,0.002995,0,0,0,0,1,1,0.379651,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.842202,1,0.57157,0.217926,0,0,0.443987,0.944883,0.909485,1,1,0.690743,0.046584;0.020086,1,1,1,0.626418,0.578633,0.039212,0,0,0.202396,0.468464,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.879832,1,0.427829,0,0,0,0.36544,0.871874,0.54047,0.895466,1,1,0.12653;0.036052,1,1,1,0.978289,1,0.351349,0.001744,0.291119,0.808102,0.792641,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0.634297,1,1,0.044682;0,0.953439,1,1,1,1,0.695814,0.122357,0.665818,0.918179,1,1,0.96905,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.995184,1,0,0,0,0,0,0,0,0.4302,1,1,0;0,1,1,1,1,1,0.788467,0.443538,0.976453,0.821757,1,1,0.91874,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.990958,1,0.015826,0.011471,0.0291,0,0,0.799206,0.039124,0.143336,1,1,0;0,0.987872,1,1,1,1,0.956499,0.846287,1,1,1,0.994108,0.803454,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.911818,1,0.255718,0.155477,0.201904,0,0,0.169106,0.012239,0.035138,1,0.992396,0;0,0.331559,0.994456,1,1,1,1,0.993464,1,1,1,0.846568,0.50482,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.678631,0.999372,0.406188,0.220154,0.351924,0.123345,0.060832,0.294028,0.228626,1,1,0.911203,0.398162;0,0,0.708728,0.989886,1,1,0.984664,0.86696,1,1,0.810749,0.594343,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.39311,0.89154,0.534804,0.363105,0.555824,0.464227,0.492411,1,1,1,0.936268,0.804315,0.092235;0,0,0.029155,0.608327,0.979577,0.972805,0.642372,0.55362,0.676798,0.661193,0.388366,0.170718,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.236419,0.667554,0.95862,1,0.880847,0.663071,0.588475,1,1,0.994936,0.896945,0.649583,0;0,0,0,0.09297,0.524128,0.56956,0.244484,0.116237,0.031457,0.18348,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1203,0.483332,0.75829,0.87546,0.864552,0.629931,0.529999,0.927271,0.937606,0.897085,0.689556,0.333384,0;0,0,0,0,0,0.127571,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.285219,0.585104,0.677819,0.663072,0.71734,0.707148,0.622624,0.643759,0.588585,0.360655,0.164273,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.022398,0.363497,0.53057,0.565147,0.583412,0.538038,0.413615,0.410336,0.408491,0.316477,0.227843,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.094155,0.260077,0.379271,0.290756,0.486927,0.484644,0.391302,0.35897,0.292468,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.207998,0.230338,0.179377,0.166829,0,0,0]; end
        case 'S+0'; name='Vagopteron'; if isLoadCell; deltaType=2;kernelType=2; R=26;peaks=[1];mu=0.218;sigma=0.0351;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.03536,0.09228,0.132929,0.157393,0.16716,0.164158,0.150399,0.127832,0.098285,0.063427,0.02473,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.082445,0.191194,0.277072,0.348764,0.401872,0.434147,0.449411,0.450599,0.439642,0.417806,0.386096,0.345507,0.297096,0.241963,0.181237,0.118211,0.06645,0.01858,0.003999,0.003453,0.001423,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.127545,0.270511,0.38874,0.480288,0.534917,0.562955,0.575238,0.590456,0.602289,0.607937,0.605929,0.594345,0.57174,0.537524,0.491869,0.435426,0.369096,0.305048,0.237762,0.165152,0.099901,0.045737,0.023887,0.015739,0.010969,0.002184,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0.085219,0.264266,0.411055,0.516122,0.559949,0.559543,0.552604,0.546489,0.545225,0.553329,0.570493,0.602237,0.630487,0.648691,0.65241,0.639461,0.609607,0.564929,0.515881,0.455148,0.383488,0.305753,0.235571,0.158364,0.094897,0.049629,0.033916,0.020501,0.010457,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0.169897,0.353617,0.494644,0.551394,0.530167,0.482339,0.427668,0.378736,0.350411,0.360745,0.39144,0.439507,0.500323,0.568348,0.625136,0.662614,0.676498,0.669578,0.647077,0.606774,0.550496,0.486545,0.420288,0.342152,0.266367,0.190813,0.120378,0.067166,0.044734,0.026633,0.013404,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0.017804,0.217021,0.402447,0.527037,0.535794,0.463225,0.369633,0.269635,0.190719,0.15409,0.140863,0.163107,0.215306,0.288421,0.37909,0.481285,0.574032,0.644234,0.688085,0.704107,0.692612,0.657577,0.612469,0.557332,0.487415,0.421457,0.345104,0.267216,0.192337,0.12198,0.069297,0.046481,0.02549,0.010725,0,0,0,0,0;0,0,0,0,0,0,0,0,0.022738,0.229439,0.421862,0.535284,0.511555,0.407931,0.275555,0.144205,0.061859,0.005595,0,0,0.033683,0.093664,0.175233,0.281289,0.403882,0.523956,0.626652,0.696626,0.727547,0.722177,0.697947,0.654971,0.597422,0.539107,0.466611,0.398885,0.321131,0.245544,0.16905,0.103199,0.06071,0.038966,0.018649,0.002395,0,0,0,0;0,0,0,0,0,0,0,0.011059,0.211434,0.414565,0.535896,0.50229,0.382085,0.223248,0.081631,0,0,0,0,0,0,0.039333,0.115996,0.221525,0.349484,0.48801,0.615255,0.706238,0.749837,0.757109,0.732787,0.687925,0.633976,0.56952,0.505094,0.429963,0.360387,0.27798,0.206564,0.130219,0.072392,0.042149,0.022047,0.007337,0,0,0,0;0,0,0,0,0,0,0,0.166519,0.385748,0.531301,0.513216,0.3919,0.215638,0.065227,0,0,0,0,0,0,0,0.041159,0.108735,0.200572,0.320663,0.464567,0.606112,0.71487,0.775489,0.791333,0.767564,0.724071,0.666341,0.602341,0.535698,0.463966,0.39019,0.312677,0.2323,0.154581,0.085073,0.041247,0.021166,0.006961,0,0,0,0;0,0,0,0,0,0,0.107019,0.336785,0.513617,0.536452,0.433035,0.250227,0.087852,0,0,0,0,0,0,0.008393,0.043318,0.085186,0.140865,0.21551,0.31722,0.448835,0.591104,0.712492,0.791464,0.818485,0.804811,0.761079,0.705425,0.64308,0.572874,0.506217,0.426494,0.351571,0.262579,0.182109,0.099374,0.03804,0.014995,0.00282,0,0,0,0;0,0,0,0,0,0.042211,0.262373,0.470841,0.555341,0.49289,0.322194,0.148665,0.011659,0,0,0,0,0.007369,0.046163,0.082519,0.117659,0.157546,0.204168,0.261024,0.337474,0.440348,0.563911,0.688624,0.782433,0.831109,0.833313,0.802379,0.752226,0.69551,0.634164,0.564258,0.493111,0.408506,0.32025,0.222401,0.12799,0.045197,0.007497,0,0,0,0,0;0,0,0,0,0.002082,0.164123,0.393301,0.54947,0.550094,0.419809,0.246805,0.091011,0,0,0,0.01098,0.050978,0.088287,0.130162,0.171825,0.209757,0.251474,0.289552,0.331525,0.387954,0.465089,0.572409,0.696049,0.809261,0.884843,0.907669,0.881894,0.81642,0.766649,0.713063,0.653103,0.581709,0.502879,0.404578,0.296304,0.182898,0.073552,0.002333,0,0,0,0,0;0,0,0,0,0.069054,0.29483,0.501534,0.57782,0.51958,0.372102,0.208228,0.086809,0.02798,0.026871,0.051341,0.091132,0.132434,0.177696,0.224981,0.273493,0.317956,0.355658,0.392862,0.425636,0.470246,0.534261,0.624309,0.742239,0.875063,0.99821,1,1,1,1,0.967566,0.824545,0.708587,0.629549,0.530193,0.406584,0.271291,0.134225,0.014223,0,0,0,0,0;0,0,0,0.007282,0.172313,0.403395,0.559471,0.589523,0.498012,0.352709,0.217738,0.129566,0.098568,0.105577,0.133764,0.175173,0.218654,0.275474,0.329499,0.378886,0.423158,0.451501,0.475705,0.493391,0.52055,0.572522,0.648882,0.75479,0.887778,1,1,1,1,1,1,1,1,0.980517,0.71252,0.567642,0.404982,0.230418,0.06625,0,0,0,0,0;0,0,0,0.059885,0.27534,0.489659,0.600481,0.587519,0.492639,0.372943,0.27141,0.208187,0.185452,0.19469,0.224114,0.264311,0.318657,0.377058,0.429876,0.467525,0.492863,0.493831,0.493316,0.500249,0.519918,0.559624,0.624047,0.688344,0.738966,0.810089,0.898966,0.99714,1,1,1,1,1,1,1,1,0.625291,0.378568,0.157502,0,0,0,0,0;0,0,0.005711,0.133588,0.35793,0.541406,0.623539,0.601617,0.521616,0.428174,0.358469,0.313643,0.294911,0.299213,0.321589,0.366287,0.418362,0.46772,0.500413,0.514012,0.496555,0.476389,0.454044,0.444306,0.46975,0.519133,0.590097,0.60005,0.61446,0.633333,0.673209,0.738163,0.826134,0.931072,1,1,1,1,1,1,1,0.7936,0.311913,0.044675,0,0,0,0;0,0,0.032023,0.20826,0.428834,0.582436,0.638538,0.620954,0.569026,0.512613,0.466195,0.437321,0.420363,0.422027,0.439879,0.470418,0.505127,0.528065,0.532473,0.502647,0.461586,0.413073,0.385231,0.386176,0.413718,0.46762,0.548498,0.523811,0.505873,0.498289,0.505352,0.532634,0.579028,0.658835,0.725768,0.825577,1,1,1,1,1,1,0.87716,0.196622,0,0,0,0;0,0,0.071489,0.268475,0.474913,0.609478,0.659338,0.654225,0.6278,0.601336,0.588495,0.580316,0.572578,0.564813,0.563892,0.570004,0.574806,0.565691,0.525794,0.47237,0.404785,0.357869,0.333494,0.336571,0.381595,0.45632,0.484173,0.457997,0.440358,0.415572,0.39321,0.386726,0.247897,0.07939,0,0.02056,0.148137,0.426182,0.803306,1,1,1,1,0.799795,0.010848,0,0,0;0,0.011403,0.112788,0.310329,0.501603,0.62337,0.674226,0.684635,0.682718,0.68567,0.700711,0.717631,0.726177,0.718829,0.698281,0.671516,0.638276,0.586064,0.525926,0.447155,0.384444,0.337087,0.321323,0.345326,0.400619,0.464378,0.436462,0.410185,0.369685,0.343842,0.312216,0.071208,0,0,0,0,0.024171,0.085218,0.173868,0.4808,1,1,1,0.981975,0.559589,0,0,0;0,0.025253,0.146509,0.340503,0.513884,0.62798,0.683087,0.707329,0.723894,0.745937,0.779398,0.820468,0.851236,0.855666,0.828822,0.779304,0.716458,0.642103,0.553928,0.479238,0.412009,0.379478,0.376507,0.415376,0.45546,0.426259,0.384815,0.344462,0.309082,0.264231,0.050539,0,0,0,0,0,0,0.044391,0.127401,0.234477,0.405266,0.926522,1,0.978586,0.907727,0.197581,0,0;0,0.042126,0.169288,0.356928,0.515076,0.624443,0.683399,0.715962,0.740952,0.770376,0.810498,0.861868,0.916921,0.950934,0.954294,0.917373,0.844872,0.748133,0.656564,0.574372,0.517723,0.497329,0.507889,0.455005,0.422651,0.370185,0.330682,0.277778,0.230493,0.118484,0,0,0,0,0,0,0,0.021726,0.100392,0.209685,0.330556,0.517059,0.905402,0.972592,0.892912,0.633659,0,0;0.002771,0.054969,0.182145,0.358173,0.506708,0.611967,0.671748,0.705198,0.733793,0.761428,0.793357,0.838774,0.913399,0.999206,1,1,0.987559,0.908923,0.795406,0.690611,0.599055,0.532581,0.464338,0.42443,0.364497,0.324406,0.25905,0.195986,0.123014,0,0,0,0,0,0,0,0,0.004702,0.082016,0.198429,0.310316,0.438886,0.545947,0.882774,0.871506,0.749295,0.158205,0;0.00864,0.063887,0.186066,0.345813,0.488977,0.588944,0.645409,0.682868,0.705586,0.726131,0.747758,0.776124,0.863835,1,1,1,1,0.969456,0.843352,0.697166,0.586994,0.491018,0.43373,0.368085,0.320507,0.250394,0.175239,0.09053,0,0,0,0,0,0,0,0,0,0,0.076308,0.200028,0.308112,0.397166,0.457608,0.538732,0.844336,0.722178,0.452007,0;0.012529,0.06895,0.181767,0.326961,0.461417,0.553954,0.610935,0.645534,0.664125,0.673704,0.690005,0.709264,0.80873,1,1,1,1,1,0.897578,0.70844,0.56052,0.46046,0.384834,0.327529,0.250812,0.1662,0.064065,0,0,0,0,0,0,0,0,0,0,0,0.085806,0.214488,0.308858,0.367096,0.389512,0.353873,0.600315,0.690275,0.568792,0;0.01424,0.070094,0.169645,0.301449,0.423505,0.506394,0.569456,0.597031,0.61291,0.621183,0.627686,0.646314,0.808834,1,1,1,1,1,0.955894,0.728893,0.540841,0.429183,0.346256,0.262058,0.164115,0.042298,0,0,0,0,0,0,0,0,0,0,0,0.005358,0.110899,0.236681,0.310782,0.331942,0.319811,0.265661,0.394838,0.656336,0.546868,0.154006;0.013525,0.067068,0.15126,0.26713,0.375086,0.462506,0.515227,0.548723,0.559266,0.562706,0.573042,0.590553,0.975495,1,1,1,1,1,1,0.756666,0.536811,0.394425,0.292032,0.177703,0.024426,0,0,0,0,0,0,0,0,0,0,0,0,0.026816,0.147052,0.259774,0.304936,0.283879,0.247104,0.188313,0.27263,0.537637,0.520905,0.281502;0.010093,0.059423,0.136083,0.229435,0.32793,0.408762,0.460828,0.489822,0.507123,0.512887,0.517864,0.669389,1,1,1,1,1,1,1,0.793565,0.545738,0.248747,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.057934,0.199003,0.283306,0.273332,0.228933,0.170483,0.121799,0.220891,0.461174,0.492421,0.338622;0.003646,0.046562,0.114618,0.195688,0.279102,0.346767,0.404141,0.434115,0.44654,0.46032,0.472704,1,1,1,1,1,1,1,0.904157,0.389352,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.006023,0.126795,0.246836,0.269491,0.225577,0.157789,0.105813,0.07446,0.233865,0.422676,0.46294,0.340941;0,0.035467,0.095239,0.163123,0.231727,0.295254,0.337876,0.375238,0.393677,0.401448,0.672779,1,1,1,1,1,1,1,0.833638,0.298317,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.059085,0.203973,0.274316,0.236118,0.152051,0.091843,0.054894,0.13306,0.28474,0.40977,0.434016,0.324888;0,0.024024,0.074356,0.132105,0.187497,0.240117,0.284163,0.310331,0.330803,0.345451,0.886482,1,1,1,1,1,1,1,0.819765,0.272863,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.03247,0.157846,0.25604,0.247032,0.166577,0.080934,0.035549,0.167875,0.279295,0.356321,0.406362,0.398818,0.282544;0,0.008545,0.046274,0.102783,0.153799,0.193804,0.228433,0.252917,0.268384,0.358842,0.752852,0.961425,1,0.82494,0.935506,1,1,1,0.836168,0.31183,0.019891,0,0,0,0,0,0,0,0,0,0,0,0,0,0.023635,0.137988,0.248198,0.253516,0.172237,0.083,0.16632,0.320624,0.406486,0.415589,0.400424,0.381199,0.340345,0.222943;0,0,0.029213,0.073187,0.115502,0.156205,0.179735,0.200318,0.206749,0.381373,0.598743,0.652127,0.485284,0.624057,0.786914,0.969632,1,1,0.87099,0.407284,0.075249,0.010156,0,0,0,0,0,0,0,0,0,0,0,0.036899,0.142308,0.246478,0.261436,0.396912,0.333317,0.391062,0.572338,0.576357,0.501141,0.43948,0.382841,0.320812,0.260876,0.151921;0,0,0.007534,0.04157,0.08447,0.115511,0.142566,0.156268,0.157327,0.230729,0.306933,0.186832,0.251548,0.421431,0.620194,0.833893,1,1,0.910635,0.535499,0.16646,0.065913,0.016953,0,0,0,0,0,0,0,0,0.01757,0.076057,0.170116,0.251738,0.290804,0.457101,0.674395,0.629525,0.650172,0.568945,0.489354,0.420593,0.3647,0.291827,0.222572,0.165518,0.073304;0,0,0,0.018199,0.045994,0.081952,0.104216,0.113959,0.119554,0.115188,0.105806,0.097146,0.099638,0.205484,0.42859,0.666786,0.893498,1,0.930517,0.665853,0.327087,0.161946,0.088549,0.042554,0.025707,0.014127,0.009285,0.012363,0.024635,0.04654,0.091368,0.15591,0.233171,0.273811,0.303847,0.49227,0.729794,0.736024,0.638803,0.553186,0.475475,0.398859,0.331605,0.253063,0.150955,0.093622,0.056858,0;0,0,0,0,0.020952,0.04317,0.066715,0.08165,0.085215,0.080532,0.071023,0.060679,0.053797,0.057906,0.218093,0.472693,0.721151,0.895392,0.91226,0.762764,0.507344,0.29932,0.200821,0.15072,0.121367,0.11219,0.115031,0.130034,0.154344,0.200072,0.243579,0.276731,0.266821,0.264138,0.4893,0.773456,0.715771,0.620675,0.533165,0.4498,0.368216,0.289801,0.158597,0.036563,0,0,0,0;0,0,0,0,0,0.016104,0.033402,0.042726,0.047923,0.046876,0.039623,0.030521,0.024548,0.021528,0.028598,0.256411,0.505746,0.722656,0.82053,0.788268,0.640815,0.464942,0.345637,0.284117,0.25159,0.240131,0.245887,0.263987,0.286877,0.295675,0.288529,0.251776,0.192025,0.415273,0.778055,0.689471,0.597862,0.510833,0.424832,0.3373,0.246076,0.02816,0,0,0,0,0,0;0,0,0,0,0,0,0.00427,0.017009,0.022895,0.02307,0.019182,0.013244,0.007454,0.003783,0.0036,0.019984,0.275368,0.495709,0.656699,0.709626,0.668953,0.569695,0.469952,0.403809,0.368779,0.351076,0.343084,0.334573,0.310933,0.27678,0.218909,0.148421,0.253049,0.637463,0.648597,0.573932,0.488362,0.402437,0.310374,0.202939,0.086054,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.025407,0.241717,0.42101,0.542673,0.586205,0.56558,0.510616,0.459755,0.414873,0.3836,0.348769,0.301186,0.247259,0.177913,0.097232,0.042089,0.352352,0.603073,0.528957,0.455801,0.374958,0.281237,0.168856,0.039986,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.158242,0.291674,0.380774,0.424866,0.425542,0.393372,0.357152,0.314229,0.257619,0.186253,0.117478,0.034903,0,0.029237,0.366395,0.492036,0.419738,0.34489,0.25998,0.154188,0.03016,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.033976,0.138085,0.200314,0.235162,0.225602,0.205072,0.161114,0.099534,0.023006,0,0,0,0,0,0.25589,0.319355,0.243833,0.153492,0.047995,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.006353,0.019166,0.008495,0,0,0,0,0,0,0,0,0,0.080879,0.158251,0.073206,0,0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.025553,0.025357,0,0,0,0,0,0,0,0,0,0,0]; end

        %otherwise; name='Orbium'; if isLoadCell; R=13;peaks=[1];mu=0.15;sigma=0.014;dt=0.1;cells=[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0;0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0;0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0;0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0;0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0;0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0;0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0;0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0;0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0;0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07;0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11;0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1;0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05;0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01;0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0;0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0;0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0;0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0;0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0;0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]; end
        otherwise; name='Orbium'; if isLoadCell; R=13;peaks=[1];mu=0.17;sigma=0.015;dt=0.1;cells=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.059332,0,0,0;0,0,0,0,0,0,0,0.003574,0.05971,0.045708,0,0,0,0,0,0.414014,0,0,0;0,0,0,0,0,0,0.094581,0.195334,0.258132,0.262641,0.193349,0.180666,0.16998,0.119761,0.059213,0.337602,0,0,0;0,0,0,0,0,0.064188,0.277279,0.348369,0.316879,0.276967,0.190642,0.130554,0.140606,0.172626,0.221208,0.212634,0.639376,0,0;0,0,0,0,0.061985,0.184252,0.293552,0.232389,0.086451,0.119372,0.074763,0,0,0,0.042733,0.193959,0.960091,0,0;0.039892,0,0,0.024903,0.183899,0.170035,0.101086,0.094981,0.041419,0.103762,0.173247,0.132177,0,0,0,0,0.505558,0.560411,0;0,0.096791,0,0.144419,0.173107,0.045669,0,0.140328,0.249356,0.21113,0.368462,0.36262,0,0,0,0,0,0.810646,0;0,0.485639,0.061916,0.207508,0.068519,0,0,0.263153,0.413258,0.434832,0.567046,0.580776,0.518581,0,0,0,0,0.45733,0.235428;0,0.200789,0.42844,0.18739,0,0,0,0.39315,0.574217,0.684601,0.739473,0.794495,0.775704,0.205565,0,0,0,0.192928,0.399319;0,0,0.876031,0.130726,0,0,0,0.418128,0.723402,0.85973,0.879693,0.972335,0.968639,0.73317,0,0,0,0.126273,0.36368;0,0,0.858375,0.119381,0,0,0,0.222769,0.863273,0.986245,0.966146,1,1,1,0.366368,0,0,0.148929,0.305449;0,0,0.348197,0.517532,0,0,0,0.078044,0.955023,1,0.930371,0.775514,0.986953,1,0.64777,0.181311,0.095203,0.206997,0.246972;0,0,0,0.704738,0,0,0,0,0.761511,1,0.924706,0.702096,0.809617,0.965154,0.723431,0.340941,0.215723,0.259242,0.177299;0,0,0,0.477571,0.243754,0,0,0,0.475688,0.979945,0.9325,0.764719,0.777214,0.825236,0.663654,0.414592,0.298108,0.256252,0.085375;0,0,0,0.084149,0.388548,0.124806,0.004186,0.057598,0.313153,0.698021,0.83963,0.785898,0.732571,0.689091,0.557382,0.39938,0.304215,0.17221,0.003563;0,0,0,0,0.219977,0.283617,0.171757,0.159214,0.26298,0.471433,0.621929,0.643042,0.606793,0.535285,0.434865,0.332035,0.213242,0.057652,0;0,0,0,0,0,0.18379,0.265813,0.250169,0.276988,0.358479,0.435022,0.454668,0.436617,0.377443,0.305736,0.200578,0.066447,0,0;0,0,0,0,0,0,0.093151,0.201859,0.250236,0.283819,0.30759,0.306075,0.269519,0.225611,0.141559,0.049351,0,0,0;0,0,0,0,0,0,0,0,0.053438,0.096974,0.120425,0.116692,0.08766,0.04753,0,0,0,0,0]; end
    end
end

function st = Rule2Str(R, peaks, mu, sigma, dt, deltaType, kernelType)
    global KERNEL_IDS DELTA_IDS
    peakSt = regexprep(strtrim(rats(peaks)), ' +', ',');
    st = sprintf('R=%d;k=%s(%s);d=%s(%g,%g)*%g', R, KERNEL_IDS{kernelType}, peakSt, DELTA_IDS{deltaType}, mu, sigma, dt);
end

function st = World2Str(world, m)
    global ZIP_START
    w = CenterWorld(world, m);
    w = floor(w * 100);
    w = UnpadWorld(w);

    c = char(w+ZIP_START-1);
    c(w==0) = '0';
    c(w==100) = '1';

    ca = cellstr(c);
    for i = 1:length(ca)
        i1 = find(w(i,:), 1, 'first');
        i2 = find(w(i,:), 1, 'last');
        lead = i1 - 1;
        if lead == 0; leadSt = '';
        elseif lead == 1; leadSt = '.';
        elseif lead < 100; leadSt = [char(lead+ZIP_START-1) '.'];
        else; leadSt = [char(floor(lead/100)+ZIP_START-1) char(mod(lead,100)+ZIP_START-1) '.'];
        end
        ca{i} = [leadSt ca{i}(i1:i2)];
    end
    st = ['(zip)' strjoin(ca, '/')];
end

function world = UnpadWorld(world)
    iSum = sum(world, 2);
    jSum = sum(world, 1);
    i1 = find(iSum, 1, 'first');
    i2 = find(iSum, 1, 'last');
    j1 = find(jSum, 1, 'first');
    j2 = find(jSum, 1, 'last');
    world = world(i1:i2, j1:j2);
end

function [R, peaks, mu, sigma, dt, deltaType, kernelType, cells] = Str2RuleAndCell(st)
    global KERNEL_IDS DELTA_IDS
    re = '^R=(?<r>.*?);k=(?<kn>.*?)(?<kp>\(.*?\))?(?<kv>\(.*?\))?;d=(?<dn>.*?)\((?<dm>.*?),(?<ds>.*?)\)\*(?<dt>.*?);(?<vn>.*?)=(?<vf>\(.*?\))?(?<v>.*)';
    parts = regexp(st, re, 'names');

    R = str2double(parts.r);
    mu = str2double(parts.dm);
    sigma = str2double(parts.ds);
    dt = str2double(parts.dt);
    deltaType = find(strcmp(DELTA_IDS, parts.dn));
    kernelType = find(strcmp(KERNEL_IDS, parts.kn));

    if isempty(parts.kp) || strcmp(parts.kp, '()')
        peaks = [1];
    else
        p1 = regexp(parts.kp, '(\d*)/?(\d*)?', 'tokens');
        p2 = str2double(vertcat(p1{:}))';
        p2(isnan(p2)) = 1;
        peaks = p2(1,:) ./ p2(2,:);
    end

    if strcmp(parts.vn, 'cells') && strcmp(parts.vf, '(zip)')
        cells = Str2Cell(parts.v);
    else
        cells = [];
    end
end

function cells = Str2Cell(st)
    global ZIP_START
    ca = strsplit(st, '/');
    maxLen = 0;
    for i = 1:length(ca)
        cs = split(ca{i}, '.');
        if length(cs)==2
            leadSt = cs{1};
            if isempty(leadSt); lead = 1;
            elseif length(leadSt) == 1; lead = int16(leadSt(1))-ZIP_START+1;
            elseif length(leadSt) == 2; lead = (int16(leadSt(1))-ZIP_START+1) * 100 + (int16(leadSt(2))-ZIP_START+1);
            end
            ca{i} = [repmat('0', 1, lead) cs{2}];
        end
        if length(ca{i}) > maxLen
            maxLen = length(ca{i});
        end
    end
    
    c = char(ca);
    c(c==' ') = '0';
    c(c=='0') = char(0+ZIP_START-1);
    c(c=='1') = char(100+ZIP_START-1);
    cells = double(c)-ZIP_START+1;
    cells = cells / 100;
end

%{
function [specs, rules] = ParseRule(st)
    st1 = '[">3", "family: A Astridae", "??? ????"],';
    st2 = '["4A4", "Tetrastrium rotans", "???(?)", "R=36;k=quad4(1,2/3,1/3,2/3);d=gaus(0.17,0.017)*0.1;cells=(zip)Ù.ÂÐàìñéÔ/Ö.ÀÑí?111111?ë/Ô.Èí?111111111???Ú/Ò.Äë111111111111??öâÓ/Ñ.Ì?1111111111111??óÕ/Ð.Ï?111111111111111??Ù/Ï.Éù11111111111???111?ï/Î.ÀØ?1111111111??îÒÅâ11?Ö/Î.ÁÔ?1111111111?ùÑ000Ã??ú/Î.Á11111111111??â00000É??å/Í.ß11111111?000000000000Ô1?Ñ/Ì.û11111?ú0000000000000000æ1?/Ë.?11?æÑÂ0000000000ÉÐ0000000?11/Ã.Â00000Á?11×0000000000000ÀÛ??ç00000À11?/Ã.É0000Û?1?Â000000000000000Êû1?Ñ00000á11ñÁÂ/Â.éÑ00Ññ?1ó0000000000000000000Þ÷Õ00000Â111ËÔÓ/Á.ÆôàÙé??1á00000000000000000000000000000?111çõá/Á.÷?÷þ?11Ô0000000000000ó1?Ù0000000000000ï111ú??â/.Æ????11é000000Â000000ï111?0000000000000Ï111111?Ù/.æ???111000000åöß00000?111?Â00000000000001111111?È/.þ11111ÿ000000??Ä00000æ1111Ý0000000000000?1111111ó/.?11111øÄ0000ã1?000000Á?111?0000ÙîÏ000000011111111Ê/Á?11111?åÌ000æúÒ0000000Õ111100Àõ111000000011111111ì/Á?11111??öØ00ÐÐ00000000Õ11111Ò?1111?000000?1111111?À/.÷11111???ù00000000000Ç111111111111?000000þ11111111Î/.ã11111111?00000000Öí??11111111111?ó000000å11111111ä/.Ð11111111?000000Ì??11111111111??íÐ0000000×?1111111û/.Â?1111111?000000è111111111111ì00000000000Ð???11111?Ä/Á.õ11111111000000Ã11111ç11111?00000000ÄÍÁ0Æó???1111?Ê/Á.Ñ111111110000000111?Ï00?111é00000000çíÄ00ÊÞ÷?11111Í/Â.?1111111å000000ÂñìÈ000á1111Ï000000Ó?ý00000Ü?11111Ë/Â.Ú1111111100000000000000?111?Â00000??Þ00000Ï11111?Â/Ã.ø1111111Å0000000000000?1111Ð00000óú000000ç111???/Ã.É?111111÷0000000000000ö1111Æ00000À000000011????ï/Ä.Ñ??ø111?0000000000000Ì?11É000000000000Â?1??ûÿ?Ë/Å.Òîçþ111Ã0000000000000000000000000000É?1?úåâðõ/Æ.ÌÓÊ111Õ000000èß0000000000000000000Ô?1?äÌÄÐçÀ/Ç.ÁÁÆ11?000000???Ò0000000000000000æ1?ôÊ000ÇÝ/Ê.ø11Î00000Ï??çÄ0000000000000Â?1?à00000Ã/Ë.?110000000ÎÐÂ0000000000Ç×ö111È/Ì.?1?0000000000000000Ô??11111/Í.?1é000000000000Ý?1111111?/Í.Ï??Ù00000Ò÷??1111111111å/Î.ç?1Í000Äë??1111111111ÛÇ/Î.Ãý?1òÂÈâÿ?1111111111?äÇ/Ï.à?111????1111111111?Ø/Ï.Íø?111111111111111?ãÀ/Ï.Ïí??1111111111111?à/Î.Ôâô??111111111111?Ó/Ï.Ü???1111111111?ÛÀ/Ñ.ò?1111111?æË/Ó.Þö?ýðÞÌÁ"],';
    re1 = '"([^"]*)"';
    specs = regexp(st, re1, 'tokens');
    re2 = '^R=(?<r>.*?);k=(?<kn>.*?)(?<kp>\(.*?\))?(?<kv>\(.*?\))?;d=(?<dn>.*?)\((?<dm>.*?),(?<ds>.*?)\)\*(?<dt>.*?);(?<vn>.*?)=(?<vf>\(.*?\))?(?<v>.*)';
    if length(specs) >= 4
        rules = regexp(specs{4}{1}, re2, 'names');
    else
        rules = struct();
    end
end
%}
