import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { HeadervisiteurComponent } from './header.component';

describe('HeaderComponent', () => {
  let component: HeadervisiteurComponent;
  let fixture: ComponentFixture<HeadervisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ HeadervisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(HeadervisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
