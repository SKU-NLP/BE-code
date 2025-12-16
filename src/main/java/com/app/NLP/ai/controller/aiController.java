package com.app.NLP.ai.controller;

import com.app.NLP.global.exception.Success;
import com.app.NLP.global.template.RspTemplate;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/workout")
@Tag(name = "ai Test API", description = "Ai 관련 테스트 api입니다.")
public class aiController {
    @GetMapping
    @Operation(method = "GET", description = "응답을 테스트 합니다.")
    public RspTemplate<?> testResponse(){
        return RspTemplate.success(Success.OK,  Success.OK.getMessage());
    }
}
